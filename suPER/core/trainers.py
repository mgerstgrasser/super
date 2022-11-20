"""
suPER trainer class.
==============================================

This file defines the distributed Trainer class for the suPER algorithm.
"""  # noqa: E501

import logging
import os
import random
from collections import deque
from typing import Callable, List, Optional, Type

import numpy as np
import torch
from gym.utils import seeding
from ray.rllib.algorithms.dqn.dqn import DQN
from ray.rllib.algorithms.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.algorithms.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.algorithms.simple_q.simple_q import SimpleQ, SimpleQConfig
from ray.rllib.execution.common import LAST_TARGET_UPDATE_TS, NUM_TARGET_UPDATES
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import DEPRECATED_VALUE, Deprecated
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.replay_buffers.utils import (
    sample_min_n_steps_from_buffer,
    update_priorities_in_replay_buffer,
)
from ray.rllib.utils.typing import AlgorithmConfigDict, ResultDict
from scipy.stats import norm

logger = logging.getLogger(__name__)


class suPERDQNConfig(SimpleQConfig):
    """Defines a configuration class from which a DQN Algorithm can be built."""

    def __init__(self, algo_class=None):
        """Initializes a DQNConfig instance."""
        super().__init__(algo_class=algo_class or suPERDQN)

        # === suPER ===
        self.suPER = True
        # Do we insert experiences as a multiagent batch, or a single-agent batch into the underlying buffer?
        # alternative: underlying_buffer
        self.suPER_insertion = "multiagent"
        # Do we use rllib's method for td error on the entire batch, or our own?
        self.suPER_td_error = "rllib"
        # alternative: 'ours'
        # Which method to use for experience selection
        self.suPER_mode = "quantile"
        # alternative: gaussian, stochastic
        self.suPER_bandwidth = 0.1
        # alternative: float in (0,1)
        # What window do we use to calculate the mean and stddev?
        self.suPER_mean_stddev_window = 1500
        # if left blank, sharing is conducted among all agents. Otherwise, filling
        # "suPER_team_sharing" will allow only agents that their policy
        # ID contains the "suPER_team_sharing" string to share.
        # Example: in environment of blue and red agents (denoted by "blue_id" and "red_id"),
        # setting "suPER_team_sharing"="blue" will allow only to blue agents to share with each other,
        # leaving red agent with no sharing capabilities, and unable to receive other agent shared experiences..
        self.suPER_team_sharing = None
        # hack to save cli arg dict to wandb
        # "cli_arg_dict": {},

        # DQN specific config settings.
        # fmt: off
        # __sphinx_doc_begin__
        self.num_atoms = 1
        self.v_min = -10.0
        self.v_max = 10.0
        self.noisy = False
        self.sigma0 = 0.5
        self.dueling = True
        self.hiddens = [256]
        self.double_q = True
        self.n_step = 1
        self.before_learn_on_batch = None
        self.training_intensity = None

        # Changes to SimpleQConfig's default:
        self.replay_buffer_config = {
            "type": "MultiAgentPrioritizedReplayBuffer",
            # Specify prioritized replay by supplying a buffer type that supports
            # prioritization, for example: MultiAgentPrioritizedReplayBuffer.
            "prioritized_replay": DEPRECATED_VALUE,
            # Size of the replay buffer. Note that if async_updates is set,
            # then each worker will have a replay buffer of this size.
            "capacity": 50000,
            "prioritized_replay_alpha": 0.6,
            # Beta parameter for sampling from prioritized replay buffer.
            "prioritized_replay_beta": 0.4,
            # Epsilon to add to the TD errors when updating priorities.
            "prioritized_replay_eps": 1e-6,
            # The number of continuous environment steps to replay at once. This may
            # be set to greater than 1 to support recurrent models.
            "replay_sequence_length": 1,
            # Whether to compute priorities on workers.
            "worker_side_prioritization": False,
        }
        # fmt: on
        # __sphinx_doc_end__

    @override(SimpleQConfig)
    def training(
        self,
        *,
        num_atoms: Optional[int] = None,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        noisy: Optional[bool] = None,
        sigma0: Optional[float] = None,
        dueling: Optional[bool] = None,
        hiddens: Optional[int] = None,
        double_q: Optional[bool] = None,
        n_step: Optional[int] = None,
        before_learn_on_batch: Callable[
            [Type[MultiAgentBatch], List[Type[Policy]], Type[int]],
            Type[MultiAgentBatch],
        ] = None,
        training_intensity: Optional[float] = None,
        replay_buffer_config: Optional[dict] = None,
        suPER: Optional[bool] = None,
        suPER_insertion: Optional[str] = None,
        suPER_td_error: Optional[str] = None,
        suPER_mode: Optional[str] = None,
        suPER_bandwidth: Optional[float] = None,
        suPER_mean_stddev_window: Optional[int] = None,
        suPER_team_sharing: Optional[str] = None,
        **kwargs,
    ) -> "suPERDQNConfig":
        """Sets the training related configuration.
        Args:
            num_atoms: Number of atoms for representing the distribution of return.
                When this is greater than 1, distributional Q-learning is used.
            v_min: Minimum value estimation
            v_max: Maximum value estimation
            noisy: Whether to use noisy network to aid exploration. This adds parametric
                noise to the model weights.
            sigma0: Control the initial parameter noise for noisy nets.
            dueling: Whether to use dueling DQN.
            hiddens: Dense-layer setup for each the advantage branch and the value
                branch
            double_q: Whether to use double DQN.
            n_step: N-step for Q-learning.
            before_learn_on_batch: Callback to run before learning on a multi-agent
                batch of experiences.
            training_intensity: The intensity with which to update the model (vs
                collecting samples from the env).
                If None, uses "natural" values of:
                `train_batch_size` / (`rollout_fragment_length` x `num_workers` x
                `num_envs_per_worker`).
                If not None, will make sure that the ratio between timesteps inserted
                into and sampled from the buffer matches the given values.
                Example:
                training_intensity=1000.0
                train_batch_size=250
                rollout_fragment_length=1
                num_workers=1 (or 0)
                num_envs_per_worker=1
                -> natural value = 250 / 1 = 250.0
                -> will make sure that replay+train op will be executed 4x asoften as
                rollout+insert op (4 * 250 = 1000).
                See: rllib/algorithms/dqn/dqn.py::calculate_rr_weights for further
                details.
            replay_buffer_config: Replay buffer config.
                Examples:
                {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentReplayBuffer",
                "capacity": 50000,
                "replay_sequence_length": 1,
                }
                - OR -
                {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 50000,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "prioritized_replay_eps": 1e-6,
                "replay_sequence_length": 1,
                }
                - Where -
                prioritized_replay_alpha: Alpha parameter controls the degree of
                prioritization in the buffer. In other words, when a buffer sample has
                a higher temporal-difference error, with how much more probability
                should it drawn to use to update the parametrized Q-network. 0.0
                corresponds to uniform probability. Setting much above 1.0 may quickly
                result as the sampling distribution could become heavily “pointy” with
                low entropy.
                prioritized_replay_beta: Beta parameter controls the degree of
                importance sampling which suppresses the influence of gradient updates
                from samples that have higher probability of being sampled via alpha
                parameter and the temporal-difference error.
                prioritized_replay_eps: Epsilon parameter sets the baseline probability
                for sampling so that when the temporal-difference error of a sample is
                zero, there is still a chance of drawing the sample.
        Returns:
            This updated AlgorithmConfig object.
        """
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        if num_atoms is not None:
            self.num_atoms = num_atoms
        if v_min is not None:
            self.v_min = v_min
        if v_max is not None:
            self.v_max = v_max
        if noisy is not None:
            self.noisy = noisy
        if sigma0 is not None:
            self.sigma0 = sigma0
        if dueling is not None:
            self.dueling = dueling
        if hiddens is not None:
            self.hiddens = hiddens
        if double_q is not None:
            self.double_q = double_q
        if n_step is not None:
            self.n_step = n_step
        if before_learn_on_batch is not None:
            self.before_learn_on_batch = before_learn_on_batch
        if training_intensity is not None:
            self.training_intensity = training_intensity
        if replay_buffer_config is not None:
            self.replay_buffer_config = replay_buffer_config
        if suPER is not None:
            self.suPER = suPER
        if suPER_insertion is not None:
            self.suPER_insertion = suPER_insertion
        if suPER_td_error is not None:
            self.suPER_td_error = suPER_td_error
        if suPER_mode is not None:
            self.suPER_mode = suPER_mode
        if suPER_bandwidth is not None:
            self.suPER_bandwidth = suPER_bandwidth
        if suPER_mean_stddev_window is not None:
            self.suPER_mean_stddev_window = suPER_mean_stddev_window
        if suPER_team_sharing is not None:
            self.suPER_team_sharing = suPER_team_sharing

        return self


def calculate_rr_weights(config: AlgorithmConfigDict) -> List[float]:
    """Calculate the round robin weights for the rollout and train steps"""
    if not config["training_intensity"]:
        return [1, 1]

    # Calculate the "native ratio" as:
    # [train-batch-size] / [size of env-rolled-out sampled data]
    # This is to set freshly rollout-collected data in relation to
    # the data we pull from the replay buffer (which also contains old
    # samples).
    native_ratio = config["train_batch_size"] / (
        config["rollout_fragment_length"]
        * config["num_envs_per_worker"]
        # Add one to workers because the local
        # worker usually collects experiences as well, and we avoid division by zero.
        * max(config["num_workers"] + 1, 1)
    )

    # Training intensity is specified in terms of
    # (steps_replayed / steps_sampled), so adjust for the native ratio.
    sample_and_train_weight = config["training_intensity"] / native_ratio
    if sample_and_train_weight < 1:
        return [int(np.round(1 / sample_and_train_weight)), 1]
    else:
        return [1, int(np.round(sample_and_train_weight))]


class suPERDQN(DQN):
    _allow_unknown_configs = True

    def __init__(self, **kwargs):
        """Initializes a Trainer instance.
        We call the parent init function using super(), without
        intervening with the DQNTrainer class flow. We use the init function
        in order to declare the TD_arr dict and sliding window length. TD_arr stores
        all previous TD-errors and the sliding window length tells us how many episodes
        should we use for the measurement of the TD-err mean and stddev.
        """
        super().__init__(**kwargs)
        if self.config.get("suPER_mode", "quantile") == "quantile" or self.config.get("suPER_mode", "quantile") == "gaussian":
            # Deterministic single-experience mode
            self._td_error_deque = {agent: deque(maxlen=self.config["suPER_mean_stddev_window"]) for agent in self.config["multiagent"]["policies"].keys()}
            self._get_samples_to_broadcast = self._get_samples_to_broadcast_single_deterministic
        elif self.config.get("suPER_mode", "quantile") == "stochastic":
            self._get_samples_to_broadcast = self._get_samples_to_broadcast_single_stochastic
        else:
            raise NotImplementedError("suPER_mode {} not implemented".format(self.config.get("suPER_mode", "quantile")))

    """suPER algorithm that broadcasts the single highest td-error transition per sample batch."""

    @override(DQN)
    def validate_config(self, config: AlgorithmConfigDict) -> None:
        # Call super's validation method.
        super().validate_config(config)

        # Update effective batch size to include n-step
        adjusted_rollout_len = max(config["rollout_fragment_length"], config["n_step"])
        config["rollout_fragment_length"] = adjusted_rollout_len

    @override(DQN)
    def get_default_policy_class(self, config: AlgorithmConfigDict) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":
            return DQNTorchPolicy
        else:
            return DQNTFPolicy

    @override(DQN)
    def training_step(self) -> ResultDict:
        """DQN training iteration function.
        Each training iteration, we:
        - Sample (MultiAgentBatch) from workers.
        - Store new samples in replay buffer.
        - Sample training batch (MultiAgentBatch) from replay buffer.
        - Learn on training batch.
        - Update remote workers' new policy weights.
        - Update target network every `target_network_update_freq` sample steps.
        - Return all collected metrics for the iteration.
        Returns:
            The results dict from executing the training iteration.
        """
        train_results = {}

        # We alternate between storing new samples and sampling and training
        store_weight, sample_and_train_weight = calculate_rr_weights(self.config)
        if self.config.get("suPER", False):
            suPER_results = []

        for _ in range(store_weight):
            # Sample (MultiAgentBatch) from workers.
            new_sample_batch = synchronous_parallel_sample(worker_set=self.workers, concat=True)

            # Update counters
            self._counters[NUM_AGENT_STEPS_SAMPLED] += new_sample_batch.agent_steps()
            self._counters[NUM_ENV_STEPS_SAMPLED] += new_sample_batch.env_steps()

            # Store new samples in replay buffer.
            self.local_replay_buffer.add(new_sample_batch)

            # Now we run our suPER experience broadcasting.
            # Note that we do this after every single sample batch is collected. If we want to broadcast less often,
            # this should be handled in self._get_samples_to_broadcast(). E.g. this could keep a buffer of the most recent
            # sample batches in order to look at the highest td-error over a longer time frame. If we wanted to do something
            # like that regularly, we could potentially also concatenate here and run self._suPER after the for-loop. For now
            # this way is easier though.
            if self.config.get("suPER", False):
                suPER_results.append(self._suPER(new_sample_batch))

        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }

        # Update target network every `target_network_update_freq` sample steps.
        cur_ts = self._counters[NUM_AGENT_STEPS_SAMPLED if self._by_agent_steps else NUM_ENV_STEPS_SAMPLED]

        if cur_ts > self.config["num_steps_sampled_before_learning_starts"]:
            for _ in range(sample_and_train_weight):
                # Sample training batch (MultiAgentBatch) from replay buffer.
                train_batch = sample_min_n_steps_from_buffer(
                    self.local_replay_buffer,
                    self.config["train_batch_size"],
                    count_by_agent_steps=self._by_agent_steps,
                )

                # Postprocess batch before we learn on it
                post_fn = self.config.get("before_learn_on_batch") or (lambda b, *a: b)
                train_batch = post_fn(train_batch, self.workers, self.config)

                # for policy_id, sample_batch in train_batch.policy_batches.items():
                #     print(len(sample_batch["obs"]))
                #     print(sample_batch.count)

                # Learn on training batch.
                # Use simple optimizer (only for multi-agent or tf-eager; all other
                # cases should use the multi-GPU optimizer, even if only using 1 GPU)
                if self.config.get("simple_optimizer") is True:
                    train_results = train_one_step(self, train_batch)
                else:
                    train_results = multi_gpu_train_one_step(self, train_batch)

                # Update replay buffer priorities.
                update_priorities_in_replay_buffer(
                    self.local_replay_buffer,
                    self.config,
                    train_batch,
                    train_results,
                )

                last_update = self._counters[LAST_TARGET_UPDATE_TS]
                if cur_ts - last_update >= self.config["target_network_update_freq"]:
                    to_update = self.workers.local_worker().get_policies_to_train()
                    self.workers.local_worker().foreach_policy_to_train(lambda p, pid: pid in to_update and p.update_target())
                    self._counters[NUM_TARGET_UPDATES] += 1
                    self._counters[LAST_TARGET_UPDATE_TS] = cur_ts

                # Update weights and global_vars - after learning on the local worker -
                # on all remote workers.
                with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                    self.workers.sync_weights(global_vars=global_vars)

            # We keep some statistics from suPER, which we everage over suPER invocations.
            # So far, we keep the fraction of samples shared.
            if self.config.get("suPER", False):
                train_results["suPER"] = {key: np.mean([result[key] for result in suPER_results]) for key in suPER_results[0]} if suPER_results else {}

        # Return all collected metrics for the iteration.
        return train_results

    def _suPER(self, new_sample_batch):
        # We first create a dict of (initially empty) lists of samples for each agent.
        # This will hold the samples that each agent wants to broadcast.
        broadcast_samples = {agent: None for agent in new_sample_batch.policy_batches}
        fraction_samples_shared = {agent: 0 for agent in new_sample_batch.policy_batches}

        # We loop through all the agents in the sample batch.
        # For each agent, we first calculate all the td-errors for the entire sample batch.
        # Then we decide which samples to broadcast, and insert those ito the broadcast_samples dict.
        for agent in new_sample_batch.policy_batches:
            agent_policy = self.workers.local_worker().policy_map[agent]
            agent_batch = new_sample_batch.policy_batches[agent].copy()

            if self.config.get("suPER_td_error", "rllib") == "rllib":
                # Calculate all td-errors for the entire batch, and convert to numpy array.
                if self.config["framework"] == "torch":
                    agent_batch["td_errors"] = np.array(agent_policy.compute_td_error(agent_batch["obs"], agent_batch["actions"], agent_batch["rewards"], agent_batch["new_obs"], agent_batch["dones"], agent_batch["weights"]).detach().cpu())
                else:
                    agent_batch["td_errors"] = np.array(agent_policy.compute_td_error(agent_batch["obs"], agent_batch["actions"], agent_batch["rewards"], agent_batch["new_obs"], agent_batch["dones"], agent_batch["weights"]))
            else:
                agent_batch["td_errors"] = np.zeros((agent_batch.count,))
                for step_index in range(agent_batch.count):
                    agent_batch["td_errors"][step_index] = self._calc_td_eror(
                        agent_batch["obs"][step_index], agent_batch["new_obs"][step_index], agent_batch["rewards"][step_index], agent_batch["dones"][step_index], agent_batch["actions"][step_index], agent_policy
                    )
            team_sharing = self.config.get("suPER_team_sharing", None)
            # if team sharing is off, share normally, if team sharing is on, check if you are
            # in the group that is permitted to share.
            if (team_sharing is None) or (team_sharing in agent):
                broadcast_samples[agent] = self._get_samples_to_broadcast(agent, agent_batch)
                fraction_samples_shared[agent] = (len(broadcast_samples[agent]) if broadcast_samples[agent] is not None else 0) / (agent_batch.count if agent_batch.count > 0 else 1)
                # We need to remove the td_errors again, so we can concatenate this directly to other sample batches that don't have td-errors calculated.
                if (broadcast_samples[agent] is not None) and ("td_errors" in broadcast_samples[agent]):
                    broadcast_samples[agent].pop("td_errors")

        # Finally, we insert all the broadcast samples into the replay buffer of all agents.
        # TODO: Is this the correct list of agents? We assume all the agents are in the sample batch, and thus also in broadcast_samples.
        # In the future we could potentially use a different list of agents, e.g. either all "policies_to_train" from the config, or a list
        # that's passed in through the config, or even a function to implement more complex scenarios (e.g. sharing within teams only).
        # TODO: For now we assume all the agents are always present in all samples. That makes it slightly easier to deal with agent_index in the batch.
        # It's unclear if agent_index is actually used for anything though.
        # ma_batch_dict = {}

        for agent in broadcast_samples:
            # if team sharing is off, share normally, if team sharing is on, check if you are
            # in the group that is permitted to share.
            if (team_sharing is None) or (team_sharing in agent):
                # Get the agent_index of this agent
                this_agent_index = new_sample_batch.policy_batches[agent]["agent_index"][0]
                # Concatenate all the samples broadcast by all the *other* agents.
                this_agent_received_samples = SampleBatch.concat_samples([broadcast_samples[other_agent] for other_agent in broadcast_samples if other_agent != agent and broadcast_samples[other_agent] is not None])

                # Adjust agent index
                if this_agent_received_samples.count > 0:
                    this_agent_received_samples["agent_index"] = np.full(this_agent_received_samples["agent_index"].shape, this_agent_index)
                    # Insert into replay buffer, after making it a MultiAgentBatch again.
                    # TODO are the weights correct when inserting?
                    # TODO is there an issue with .add_batch()? If yes, use .add_to_underlying_buffer() instead.
                    # ma_batch_dict[agent] = this_agent_received_samples
                    ma_batch = MultiAgentBatch({agent: this_agent_received_samples}, this_agent_received_samples.count)
                    if self.config.get("suPER_insertion", "multiagent") == "multiagent":
                        self.local_replay_buffer.add(ma_batch)
                    else:
                        self.local_replay_buffer._add_to_underlying_buffer(agent, this_agent_received_samples)
                # ma_batch = MultiAgentBatch(ma_batch_dict, this_agent_received_samples.count)
                # self.local_replay_buffer.add_batch(ma_batch)
        results = {f"fraction_samples_shared_{agent}": fraction_samples_shared[agent] for agent in fraction_samples_shared}
        results["fraction_samples_shared_mean"] = np.mean(list(fraction_samples_shared.values()))
        return results

    def _get_samples_to_broadcast_single_stochastic(self, agent_id, agent_batch):
        """Given an agent batch with td-errors already calculate, decide which transitions to broadcast.
        Here we sample transitions weighted by td-error, such that we average the configured bandwidth.

        Args:
            agent_id (str): agent ID string
            agent_batch (SampleBatch): batch of samples from an agent

        Returns:
            SampleBatch: batch of samples to broadcast
        """
        # First figure out how many samples to broadcast - for randomised it's easy, just bandwidth * batch size.
        num_samples_to_broadcast = self.config.get("suPER_bandwidth", 0.01) * len(agent_batch)
        total_td_error = np.sum(np.abs(agent_batch["td_errors"]))
        batches = []
        for i in range(len(agent_batch)):
            # The probability of sampling this transition is proportional to the td-error, divided by the total td_error,
            # and multiplied by the number of samples we would like to broadcast.
            # TODO this is actually not entirely correct - we truncate probabilities to 1.0, i.e. if there are very-high-td-error transitions,
            # we will end up with slightly lower total probability than we were aiming for, i.e. lower in-expectation bandwidth.
            p = np.min([1.0, num_samples_to_broadcast * np.abs(agent_batch["td_errors"][i]) / total_td_error])
            if np.isnan(p) or p < 0:
                p = 0.0
            if np.random.binomial(1, p) == 1:
                batches.append(agent_batch.slice(i, i + 1))
        if len(batches) > 0:
            return SampleBatch.concat_samples(batches)
        return None

    def _get_samples_to_broadcast_single_deterministic(self, agent_id, agent_batch):
        """Given an agent batch with td-errors already calculate, decide which transitions to broadcast.
        Here we broadcast any transition with td-error >= mean + c * std, where we choose c such that the expected number of
        transitions with td-error >= mean + c * std is the configured bandwidth.

        Args:
            agent_id (str): agent ID string
            agent_batch (SampleBatch): batch of samples from an agent

        Returns:
            SampleBatch: batch of samples to broadcast
        """
        # Add sample batch td errors to the running td error accumulator.
        self._td_error_deque[agent_id].extend(np.abs(agent_batch["td_errors"]))
        mean = np.mean(self._td_error_deque[agent_id])
        var = np.var(self._td_error_deque[agent_id])
        if self.config.get("suPER_mode", "quantile") == "gaussian":
            # We use the "inverse survival function" to figure out what the constant c needs to be such that (for normally dist. td-errors),
            # the expected number of transitions with td-error >= mean + c * std is the configured bandwidth.
            c = norm.isf(self.config.get("suPER_bandwidth", 0.01), loc=mean, scale=var)
        else:
            # Alternatively, we could use a percentile of the actual sampled td-errors:
            c = np.percentile(self._td_error_deque[agent_id], (1 - self.config.get("suPER_bandwidth", 0.01)) * 100)
        batches = []
        for i in range(len(agent_batch)):
            if np.abs(agent_batch["td_errors"][i]) > c:
                batches.append(agent_batch.slice(i, i + 1))
        if len(batches) > 0:
            return SampleBatch.concat_samples(batches)
        return None
