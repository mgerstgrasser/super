import os
import pickle
import random
from copy import deepcopy

import gym
import numpy as np
import torch
from gym import spaces
from gym.envs.registration import register
from gym.spaces import Tuple
from gym.spaces.discrete import Discrete
from gym.utils import seeding
from ray import air, tune
from ray.rllib.algorithms.apex_dqn.apex_dqn import ApexDQN
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.dqn import DQNTorchPolicy
from ray.rllib.algorithms.maddpg import MADDPG
from ray.rllib.algorithms.maddpg.maddpg import MADDPGConfig
from ray.rllib.algorithms.qmix import QMix
from ray.rllib.algorithms.qmix.qmix import QMixConfig
from ray.rllib.algorithms.qmix.qmix_policy import QMixTorchPolicy
from ray.rllib.algorithms.registry import get_algorithm_class
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils import merge_dicts

from suPER.core.trainers import suPERDQN, suPERDQNConfig
from suPER.experiments.env import *
from suPER.experiments.models import (
    AdversarialPursuitModel,
    AdversarialPursuitModelUnflat,
    BattleModel,
    BattleModelUnflat,
    PursuitModel,
)

# from social_dilemmas.envs.env_creator import get_env_creator


_PURSUIT_N_TIMESTEPS = 800000
_BATTLE_N_TIMESTEPS = 300000
_ADVPURSUIT_N_TIMESTEPS = 600000
_EVAL_RESOLUTION = 200

"""
##################################
Pursuit config functions
for DQN,QMIX and MADDPG.
##################################
"""


def config_pursuit_dqn(args):
    # Register custom model for Pursuit environment.
    ModelCatalog.register_custom_model("PursuitModel", PursuitModel)

    # if fully connected is true, use fully connected NN instead of conv based network.
    if args.fully_connected == "true":
        model_config = None
    else:
        model_config = {
            "model": {
                "custom_model": "PursuitModel",
            },
            "gamma": 0.99,
        }
    env_config = {"num_agents": 8, "n_evaders": 30, "shared_reward": False}

    # Overwrite env_config variables using CLI args, if given.
    if args.env_pursuit_num_agents is not None:
        env_config["num_agents"] = args.env_pursuit_num_agents
    if args.env_pursuit_n_evaders is not None:
        env_config["n_evaders"] = args.env_pursuit_n_evaders
    if args.env_pursuit_shared_reward is not None:
        env_config["shared_reward"] = args.env_pursuit_shared_reward

    if args.parameter_sharing == "true":
        # if parameter_sharing=True, only one policy is used for all agents.
        policies = {"pursuer_0": PolicySpec(None, None, None, model_config)}
    else:
        policies = {f"pursuer_{i}": PolicySpec(None, None, None, model_config) for i in range(env_config["num_agents"])}

    config = deepcopy(suPERDQNConfig().to_dict())  # deepcopy(get_trainer_class(alg_name).get_default_config())

    config["env"] = "pursuit"

    config["env_config"] = env_config

    config["num_gpus"] = 0
    config["framework"] = "torch"
    # config["log_level"] = "DEBUG"
    config["replay_buffer_config"]["capacity"] = 120000
    config["exploration_config"] = {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 0.1,
        "final_epsilon": 0.001,
        "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
    }
    config["replay_buffer_config"]["type"] = "MultiAgentPrioritizedReplayBuffer"
    config["replay_buffer_config"]["prioritized_replay_alpha"] = 0.6
    config["replay_buffer_config"]["prioritized_replay_beta"] = 0.4
    config["replay_buffer_config"]["prioritized_replay_eps"] = 1e-06
    config["train_batch_size"] = env_config["num_agents"] * 32  # each agent's sample batch is with size 32
    config["lr"] = 0.00016
    config["horizon"] = 500
    config["dueling"] = True
    config["target_network_update_freq"] = 1000
    config["rollout_fragment_length"] = 4
    config["no_done_at_end"] = False

    config["min_sample_timesteps_per_iteration"] = _PURSUIT_N_TIMESTEPS / _EVAL_RESOLUTION
    config["min_time_s_per_iteration"] = 0
    config["evaluation_interval"] = 1
    config["evaluation_duration"] = 20
    config["evaluation_duration_unit"] = "episodes"
    config["metrics_num_episodes_for_smoothing"] = 20

    config["num_steps_sampled_before_learning_starts"] = 1000
    config["multiagent"] = {"policies": policies}

    if args.parameter_sharing == "true":
        # if parameter_sharing=True, map all agents into a single policy.
        config["multiagent"]["policy_mapping_fn"] = lambda agent_id, episode, **kwargs: "pursuer_0"
    else:
        config["multiagent"]["policy_mapping_fn"] = lambda agent_id, episode, **kwargs: agent_id

    # if fully connected is true, use fully connected NN instead of conv based network.
    if args.fully_connected == "true":
        config["model"] = {"fcnet_hiddens": [64, 64]}
        config["env_config"]["flatten_obs"] = True

    stop = {"timesteps_total": _PURSUIT_N_TIMESTEPS}

    return suPERDQN, config, stop


def config_pursuit_plaindqn(args):
    # Register custom model for Pursuit environment.
    ModelCatalog.register_custom_model("PursuitModel", PursuitModel)

    # if fully connected is true, use fully connected NN instead of conv based network.
    if args.fully_connected == "true":
        model_config = None
    else:
        model_config = {
            "model": {
                "custom_model": "PursuitModel",
            },
            "gamma": 0.99,
        }
    env_config = {"num_agents": 8, "n_evaders": 30, "shared_reward": False}

    # Overwrite env_config variables using CLI args, if given.
    if args.env_pursuit_num_agents is not None:
        env_config["num_agents"] = args.env_pursuit_num_agents
    if args.env_pursuit_n_evaders is not None:
        env_config["n_evaders"] = args.env_pursuit_n_evaders
    if args.env_pursuit_shared_reward is not None:
        env_config["shared_reward"] = args.env_pursuit_shared_reward

    if args.parameter_sharing == "true":
        # if parameter_sharing=True, only one policy is used for all agents.
        policies = {"pursuer_0": PolicySpec(None, None, None, model_config)}
    else:
        policies = {f"pursuer_{i}": PolicySpec(None, None, None, model_config) for i in range(env_config["num_agents"])}

    config = deepcopy(suPERDQNConfig().to_dict())  # deepcopy(get_trainer_class(alg_name).get_default_config())

    config["env"] = "pursuit"

    config["env_config"] = env_config

    config["num_gpus"] = 0
    config["framework"] = "torch"
    # config["log_level"] = "DEBUG"
    config["replay_buffer_config"]["capacity"] = 120000
    config["exploration_config"] = {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 0.1,
        "final_epsilon": 0.001,
        "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
    }
    config["replay_buffer_config"]["type"] = "MultiAgentPrioritizedReplayBuffer"
    config["replay_buffer_config"]["prioritized_replay_alpha"] = 0.6
    config["replay_buffer_config"]["prioritized_replay_beta"] = 0.4
    config["replay_buffer_config"]["prioritized_replay_eps"] = 1e-06
    config["train_batch_size"] = env_config["num_agents"] * 32  # each agent's sample batch is with size 32
    config["lr"] = 0.00016
    config["horizon"] = 500
    config["dueling"] = False
    config["double_q"] = False
    config["target_network_update_freq"] = 1000
    config["rollout_fragment_length"] = 4
    config["no_done_at_end"] = False

    config["min_sample_timesteps_per_iteration"] = _PURSUIT_N_TIMESTEPS / _EVAL_RESOLUTION
    config["min_time_s_per_iteration"] = 0
    config["evaluation_interval"] = 1
    config["evaluation_duration"] = 20
    config["evaluation_duration_unit"] = "episodes"
    config["metrics_num_episodes_for_smoothing"] = 20

    config["num_steps_sampled_before_learning_starts"] = 1000
    config["multiagent"] = {"policies": policies}

    if args.parameter_sharing == "true":
        # if parameter_sharing=True, map all agents into a single policy.
        config["multiagent"]["policy_mapping_fn"] = lambda agent_id, episode, **kwargs: "pursuer_0"
    else:
        config["multiagent"]["policy_mapping_fn"] = lambda agent_id, episode, **kwargs: agent_id

    # if fully connected is true, use fully connected NN instead of conv based network.
    if args.fully_connected == "true":
        config["model"] = {"fcnet_hiddens": [64, 64]}
        config["env_config"]["flatten_obs"] = True

    stop = {"timesteps_total": _PURSUIT_N_TIMESTEPS}

    return suPERDQN, config, stop


def config_pursuit_maddpg(args):

    env_config = {"num_agents": 8, "n_evaders": 30, "shared_reward": True, "flatten_obs": True, "actions_are_logits": True}

    # Overwrite env_config variables using CLI args, if given.
    if args.env_pursuit_num_agents is not None:
        env_config["num_agents"] = args.env_pursuit_num_agents
    if args.env_pursuit_n_evaders is not None:
        env_config["n_evaders"] = args.env_pursuit_n_evaders
    if args.env_pursuit_shared_reward is not None:
        env_config["shared_reward"] = args.env_pursuit_shared_reward

    policies = {f"pursuer_{i}": PolicySpec(config={"agent_id": i}) for i in range(env_config["num_agents"])}

    config = deepcopy(MADDPGConfig().to_dict())  # deepcopy(get_trainer_class(alg_name).get_default_config())

    config["env"] = "pursuit"

    config["env_config"] = env_config

    config["num_gpus"] = 0
    config["framework"] = "tf"
    config["exploration_config"] = {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 0.1,
        "final_epsilon": 0.001,
        "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
    }
    config["replay_buffer_config"]["capacity"] = 120000
    config["replay_buffer_config"]["learning_starts"] = 0

    config["horizon"] = 500

    config["rollout_fragment_length"] = 32
    config["no_done_at_end"] = False

    config["critic_hiddens"] = [64, 64]
    config["actor_hiddens"] = [64, 64]

    # if set to True, training is performed with gridserch (multiple experiments)
    if args.parameter_tunning is not None:
        config["actor_feature_reg"] = tune.loguniform(0.0005, 0.002)
        config["critic_lr"] = config["actor_lr"] = tune.loguniform(0.0001, 0.001)
        config["train_batch_size"] = tune.choice([32, 64, 128, 256, 1024])
        config["tau"] = tune.loguniform(0.01, 0.2)
        stop = {"timesteps_total": 500000}
    else:  # standard training
        config["actor_feature_reg"] = 0.001
        config["critic_lr"] = config["actor_lr"] = 0.00025
        config["train_batch_size"] = 32
        config["tau"] = 0.015
        stop = {"timesteps_total": _PURSUIT_N_TIMESTEPS}

    config["min_sample_timesteps_per_iteration"] = _PURSUIT_N_TIMESTEPS / _EVAL_RESOLUTION
    config["min_time_s_per_iteration"] = 0
    config["evaluation_interval"] = 1
    config["evaluation_duration"] = 20
    config["evaluation_duration_unit"] = "episodes"
    config["metrics_num_episodes_for_smoothing"] = 20

    config["multiagent"] = {"policies": policies, "policy_mapping_fn": (lambda agent_id, episode, **kwargs: agent_id)}

    return MADDPG, config, stop


def config_pursuit_qmix(args):
    # Register custom model for Pursuit environment.
    ModelCatalog.register_custom_model("PursuitModel", PursuitModel)

    env_config = {"num_agents": 8, "n_evaders": 30, "shared_reward": False, "group_agents": True}

    # Overwrite env_config variables using CLI args, if given.
    if args.env_pursuit_num_agents is not None:
        env_config["num_agents"] = args.env_pursuit_num_agents
    if args.env_pursuit_n_evaders is not None:
        env_config["n_evaders"] = args.env_pursuit_n_evaders
    if args.env_pursuit_shared_reward is not None:
        env_config["shared_reward"] = args.env_pursuit_shared_reward

    policies = {"group_1": PolicySpec(policy_class=QMixTorchPolicy)}

    config = deepcopy(QMixConfig().to_dict())  # deepcopy(get_trainer_class(alg_name).get_default_config())

    config["env"] = "pursuit"

    config["env_config"] = env_config
    config["simple_optimizer"] = True
    config["num_gpus"] = 0
    config["framework"] = "torch"
    # config["log_level"] = "DEBUG"
    # config['replay_buffer_config']['capacity'] = 120000
    config["exploration_config"] = {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 0.1,
        "final_epsilon": 0.001,
        "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
    }
    config["replay_buffer_config"]["type"] = "MultiAgentPrioritizedReplayBuffer"
    config["replay_buffer_config"]["prioritized_replay_alpha"] = 0.6
    config["replay_buffer_config"]["prioritized_replay_beta"] = 0.4
    config["replay_buffer_config"]["prioritized_replay_eps"] = 1e-06

    config["horizon"] = 500
    config["target_network_update_freq"] = 1000
    config["rollout_fragment_length"] = 4
    config["no_done_at_end"] = False

    config["min_sample_timesteps_per_iteration"] = _PURSUIT_N_TIMESTEPS / _EVAL_RESOLUTION
    config["min_time_s_per_iteration"] = 0
    config["evaluation_interval"] = 1
    config["evaluation_duration"] = 20
    config["evaluation_duration_unit"] = "episodes"
    config["metrics_num_episodes_for_smoothing"] = 20

    if args.parameter_tunning is not None:
        config["lr"] = tune.grid_search([0.0001, 0.0002, 0.00025, 0.0003])
        config["train_batch_size"] = tune.grid_search([32, 64, 128, 256])
    else:
        config["lr"] = 0.00016
        config["train_batch_size"] = 32

    # config["num_steps_sampled_before_learning_starts"] = 1000
    config["multiagent"] = {"policies": policies, "policy_mapping_fn": (lambda agent_id, episode, **kwargs: "group_1")}

    stop = {"timesteps_total": _PURSUIT_N_TIMESTEPS}

    return QMix, config, stop


"""
##################################
Battle config functions
for DQN,QMIX and MADDPG.
##################################
"""


def policy_mapping_fn_battle(agent_id, episode, worker, **kwargs):
    team = agent_id.split("_")[0]
    if team == "blue":  # blue team is using parameter sharing
        return "blue_"
    else:
        return agent_id  # red team is not using parameter sharing


class BattleCallback(DefaultCallbacks):
    def on_algorithm_init(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,
    ) -> None:
        """Loads weights from a checkpoint, and sets the algorithm's weights to the loaded weights for red team only."""
        checkpoint_file = os.path.dirname(os.path.realpath(__file__)) + "/checkpoints/battle.weights"
        with open(checkpoint_file, "rb") as f:
            weights = pickle.load(f)
        algorithm.set_weights(weights)


def my_custom_dqn(
    observation_space,
    action_space,
    config,
):
    config["hiddens"] = [256]
    config["dueling"] = True
    config["double_q"] = True
    config["model"] = {
        "_use_default_native_models": False,
        "_disable_preprocessor_api": False,
        "_disable_action_flattening": False,
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
        "conv_filters": None,
        "conv_activation": "relu",
        "post_fcnet_hiddens": [],
        "post_fcnet_activation": "relu",
        "free_log_std": False,
        "no_final_linear": False,
        "vf_share_layers": True,
        "use_lstm": False,
        "max_seq_len": 20,
        "lstm_cell_size": 256,
        "lstm_use_prev_action": False,
        "lstm_use_prev_reward": False,
        "_time_major": False,
        "use_attention": False,
        "attention_num_transformer_units": 1,
        "attention_dim": 64,
        "attention_num_heads": 1,
        "attention_head_dim": 32,
        "attention_memory_inference": 50,
        "attention_memory_training": 50,
        "attention_position_wise_mlp_dim": 32,
        "attention_init_gru_gate_bias": 2.0,
        "attention_use_n_prev_actions": 0,
        "attention_use_n_prev_rewards": 0,
        "framestack": True,
        "dim": 84,
        "grayscale": False,
        "zero_mean": True,
        "custom_model": "BattleModel",
        "custom_model_config": {},
        "custom_action_dist": None,
        "custom_preprocessor": None,
        "lstm_use_prev_action_reward": -1,
    }
    return DQNTorchPolicy(observation_space, action_space, config)


def my_custom_dqn_unflat(
    observation_space,
    action_space,
    config,
):
    config["hiddens"] = [256]
    config["dueling"] = True
    config["double_q"] = True
    config["model"] = {
        "_use_default_native_models": False,
        "_disable_preprocessor_api": False,
        "_disable_action_flattening": False,
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
        "conv_filters": None,
        "conv_activation": "relu",
        "post_fcnet_hiddens": [],
        "post_fcnet_activation": "relu",
        "free_log_std": False,
        "no_final_linear": False,
        "vf_share_layers": True,
        "use_lstm": False,
        "max_seq_len": 20,
        "lstm_cell_size": 256,
        "lstm_use_prev_action": False,
        "lstm_use_prev_reward": False,
        "_time_major": False,
        "use_attention": False,
        "attention_num_transformer_units": 1,
        "attention_dim": 64,
        "attention_num_heads": 1,
        "attention_head_dim": 32,
        "attention_memory_inference": 50,
        "attention_memory_training": 50,
        "attention_position_wise_mlp_dim": 32,
        "attention_init_gru_gate_bias": 2.0,
        "attention_use_n_prev_actions": 0,
        "attention_use_n_prev_rewards": 0,
        "framestack": True,
        "dim": 84,
        "grayscale": False,
        "zero_mean": True,
        "custom_model": "BattleModelUnflat",
        "custom_model_config": {},
        "custom_action_dist": None,
        "custom_preprocessor": None,
        "lstm_use_prev_action_reward": -1,
    }
    return DQNTorchPolicy(observation_space, action_space, config)


def config_battlev4_dqn(args):

    env_config = {"map_size": 18}

    # Register custom model for Pursuit environment.
    ModelCatalog.register_custom_model("BattleModel", BattleModel)

    # if fully connected is true, use fully connected NN instead of conv based network.
    if args.fully_connected == "true":
        model_config = None
    else:
        model_config = {
            "model": {
                "custom_model": "BattleModel",
            },
            "gamma": 0.99,
        }

    # Overwrite env_config variables using CLI args, if given.
    if args.env_battle_map_size is not None:
        env_config["map_size"] = args.env_battle_map_size

    env = battle_v4.parallel_env(map_size=env_config["map_size"], minimap_mode=False, step_reward=-0.005, dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2, max_cycles=1000, extra_features=False)

    num_battle_agents = len(env.agents)
    policies_blue_battle = ["blue_{}".format(i) for i in range(int(num_battle_agents / 2))]
    policies_red_battle = ["red_{}".format(i) for i in range(int(num_battle_agents / 2))]
    policies_list = policies_blue_battle + policies_red_battle

    if args.parameter_sharing == "true":
        # if parameter_sharing=True, only one policy is used for all agents.
        policies = {"red_{}".format(i): PolicySpec(None, None, None, model_config) for i in range(int(num_battle_agents / 2))}
        policies_blue = {"blue_": PolicySpec(None, None, None, model_config)}
        policies.update(policies_blue)
    else:
        policies = {i: PolicySpec(None, spaces.Box(0.0, 2.0, (845,) if args.fully_connected == "true" else (13, 13, 5), np.float32), Discrete(21), model_config) for i in policies_list}

    config = deepcopy(suPERDQNConfig().to_dict())  # deepcopy(get_trainer_class(alg_name).get_default_config())

    config["env"] = "battle"
    config["env_config"] = env_config
    config["store_buffer_in_checkpoints"] = False
    config["num_gpus"] = 0
    config["framework"] = "torch"
    # config["log_level"] = "DEBUG"
    config["replay_buffer_config"]["capacity"] = 90000
    config["exploration_config"] = {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 0.1,
        "final_epsilon": 0.001,
        "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
    }
    config["replay_buffer_config"]["type"] = "MultiAgentPrioritizedReplayBuffer"
    config["rollout_fragment_length"] = 5
    config["prioritized_replay_alpha"] = 0.6
    config["prioritized_replay_beta"] = 0.4
    config["prioritized_replay_eps"] = 0.00001
    config["train_batch_size"] = num_battle_agents * 32  # each agent's sample batch is with size 32
    config["replay_buffer_config"]["worker_side_prioritization"] = True
    config["lr"] = 1e-4
    config["horizon"] = 1000
    config["dueling"] = True
    config["target_network_update_freq"] = 1200
    config["no_done_at_end"] = False
    config["num_steps_sampled_before_learning_starts"] = 1000
    config["multiagent"] = {"policies": policies}

    if args.parameter_sharing == "true":
        # if parameter_sharing=True, map all agents from the same team into a single policy.
        config["multiagent"]["policy_mapping_fn"] = policy_mapping_fn_battle
    else:
        config["multiagent"]["policy_mapping_fn"] = lambda agent_id, episode, **kwargs: agent_id

    config["min_sample_timesteps_per_iteration"] = _BATTLE_N_TIMESTEPS / _EVAL_RESOLUTION
    config["min_time_s_per_iteration"] = 0
    config["evaluation_interval"] = 1
    config["evaluation_duration"] = 20
    config["evaluation_duration_unit"] = "episodes"
    config["metrics_num_episodes_for_smoothing"] = 20

    config["suPER_team_sharing"] = "blue"

    # if fully connected is true, use fully connected NN instead of conv based network.
    if args.fully_connected == "true":
        config["model"] = {"fcnet_hiddens": [64, 64]}
        config["env_config"]["flatten_obs"] = True

    if args.battle_advpursuit_against_pretrained is True:
        ModelCatalog.register_custom_model("BattleModel", BattleModel)
        config["callbacks"] = BattleCallback
        config["multiagent"]["policies_to_train"] = [pol for pol in config["multiagent"]["policies"] if pol[:4] == "blue"]
        # config["multiagent"]["policies_to_train"] = ["group_1"]
        # model_config = {
        #     "model": {
        #         "custom_model": "BattleModel",
        #     },
        #     "gamma": 0.99,
        # }
        for pol in config["multiagent"]["policies"]:
            if pol[:3] == "red":
                config["multiagent"]["policies"][pol] = PolicySpec(my_custom_dqn, spaces.Box(0.0, 2.0, (845,) if args.fully_connected == "true" else (13, 13, 5), np.float32), spaces.Discrete(21), model_config)
        config["simple_optimizer"] = False

    stop = {"timesteps_total": _BATTLE_N_TIMESTEPS}

    return suPERDQN, config, stop


def config_battlev4_plaindqn(args):

    env_config = {"map_size": 18}

    # Register custom model for Pursuit environment.
    ModelCatalog.register_custom_model("BattleModel", BattleModel)

    # if fully connected is true, use fully connected NN instead of conv based network.
    if args.fully_connected == "true":
        model_config = None
    else:
        model_config = {
            "model": {
                "custom_model": "BattleModel",
            },
            "gamma": 0.99,
        }

    # Overwrite env_config variables using CLI args, if given.
    if args.env_battle_map_size is not None:
        env_config["map_size"] = args.env_battle_map_size

    env = battle_v4.parallel_env(map_size=env_config["map_size"], minimap_mode=False, step_reward=-0.005, dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2, max_cycles=1000, extra_features=False)

    num_battle_agents = len(env.agents)
    policies_blue_battle = ["blue_{}".format(i) for i in range(int(num_battle_agents / 2))]
    policies_red_battle = ["red_{}".format(i) for i in range(int(num_battle_agents / 2))]
    policies_list = policies_blue_battle + policies_red_battle

    if args.parameter_sharing == "true":
        # if parameter_sharing=True, only one policy is used for all agents.
        policies = {"red_{}".format(i): PolicySpec(None, None, None, model_config) for i in range(int(num_battle_agents / 2))}
        policies_blue = {"blue_": PolicySpec(None, None, None, model_config)}
        policies.update(policies_blue)
    else:
        policies = {i: PolicySpec(None, spaces.Box(0.0, 2.0, (845,) if args.fully_connected == "true" else (13, 13, 5), np.float32), Discrete(21), model_config) for i in policies_list}

    config = deepcopy(suPERDQNConfig().to_dict())  # deepcopy(get_trainer_class(alg_name).get_default_config())
    config["env"] = "battle"
    config["env_config"] = env_config
    config["store_buffer_in_checkpoints"] = False
    config["num_gpus"] = 0
    config["framework"] = "torch"
    # config["log_level"] = "DEBUG"
    config["replay_buffer_config"]["capacity"] = 90000
    config["exploration_config"] = {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 0.1,
        "final_epsilon": 0.001,
        "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
    }
    config["replay_buffer_config"]["type"] = "MultiAgentPrioritizedReplayBuffer"
    config["rollout_fragment_length"] = 5
    config["prioritized_replay_alpha"] = 0.6
    config["prioritized_replay_beta"] = 0.4
    config["prioritized_replay_eps"] = 0.00001
    config["train_batch_size"] = num_battle_agents * 32  # each agent's sample batch is with size 32
    config["replay_buffer_config"]["worker_side_prioritization"] = True
    config["lr"] = 1e-4
    config["horizon"] = 1000
    config["dueling"] = False
    config["double_q"] = False
    config["target_network_update_freq"] = 1200
    config["no_done_at_end"] = False
    config["num_steps_sampled_before_learning_starts"] = 1000
    config["multiagent"] = {"policies": policies}

    if args.parameter_sharing == "true":
        # if parameter_sharing=True, map all agents from the same team into a single policy.
        config["multiagent"]["policy_mapping_fn"] = policy_mapping_fn_battle
    else:
        config["multiagent"]["policy_mapping_fn"] = lambda agent_id, episode, **kwargs: agent_id

    config["min_sample_timesteps_per_iteration"] = _BATTLE_N_TIMESTEPS / _EVAL_RESOLUTION
    config["min_time_s_per_iteration"] = 0
    config["evaluation_interval"] = 1
    config["evaluation_duration"] = 20
    config["evaluation_duration_unit"] = "episodes"
    config["metrics_num_episodes_for_smoothing"] = 20

    config["suPER_team_sharing"] = "blue"

    # if fully connected is true, use fully connected NN instead of conv based network.
    if args.fully_connected == "true":
        config["model"] = {"fcnet_hiddens": [64, 64]}
        config["env_config"]["flatten_obs"] = True

    if args.battle_advpursuit_against_pretrained is True:
        ModelCatalog.register_custom_model("BattleModel", BattleModel)
        config["callbacks"] = BattleCallback
        config["multiagent"]["policies_to_train"] = [pol for pol in config["multiagent"]["policies"] if pol[:4] == "blue"]
        for pol in config["multiagent"]["policies"]:
            if pol[:3] == "red":
                config["multiagent"]["policies"][pol] = PolicySpec(my_custom_dqn, spaces.Box(0.0, 2.0, (845,) if args.fully_connected == "true" else (13, 13, 5), np.float32), spaces.Discrete(21), model_config)
        config["simple_optimizer"] = False

    stop = {"timesteps_total": _BATTLE_N_TIMESTEPS}

    return suPERDQN, config, stop


def config_battlev4_maddpg(args):
    # Only do MADDPG against pretrained, this is hardcoded in env now.
    assert args.battle_advpursuit_against_pretrained is True

    env_config = {"map_size": 18, "flatten_obs": True, "actions_are_logits": True}

    # Overwrite env_config variables using CLI args, if given.
    if args.env_battle_map_size is not None:
        env_config["map_size"] = args.env_battle_map_size

    env = battle_v4.parallel_env(map_size=env_config["map_size"], minimap_mode=False, step_reward=-0.005, dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2, max_cycles=1000, extra_features=False)

    num_battle_agents = len(env.agents)
    policies_blue_battle = ["blue_{}".format(i) for i in range(int(num_battle_agents / 2))]
    policies_red_battle = ["red_{}".format(i) for i in range(int(num_battle_agents / 2))]
    policies_list = policies_blue_battle + policies_red_battle
    policies = {i: PolicySpec(None, None, None, config={"agent_id": count}) for count, i in enumerate(policies_list)}
    config = deepcopy(MADDPGConfig().to_dict())  # deepcopy(get_trainer_class(alg_name).get_default_config())

    config["env"] = "battle"

    config["env_config"] = env_config

    config["num_gpus"] = 0
    config["framework"] = "tf"
    config["exploration_config"] = {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 0.1,
        "final_epsilon": 0.001,
        "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
    }

    config["replay_buffer_config"]["capacity"] = 90000
    config["replay_buffer_config"]["learning_starts"] = 0

    config["horizon"] = 1000

    config["rollout_fragment_length"] = 32
    config["no_done_at_end"] = False

    config["critic_hiddens"] = [64, 64]
    config["actor_hiddens"] = [64, 64]

    # if set to True, training is performed with gridserch (multiple experiments)
    if args.parameter_tunning is not None:
        config["actor_feature_reg"] = tune.grid_search([0.001, 0.0015, 0.002])
        config["critic_lr"] = config["actor_lr"] = tune.grid_search([0.0001, 0.0002, 0.00025, 0.0003])
        config["train_batch_size"] = tune.grid_search([32, 64, 128, 256])
        config["tau"] = tune.grid_search([0.05, 0.01, 0.015, 0.2])
    else:  # standard training
        config["actor_feature_reg"] = 0.001
        config["critic_lr"] = config["actor_lr"] = 0.00025
        config["train_batch_size"] = 32
        config["tau"] = 0.015

    config["min_sample_timesteps_per_iteration"] = _BATTLE_N_TIMESTEPS / _EVAL_RESOLUTION
    config["min_time_s_per_iteration"] = 0
    config["evaluation_interval"] = 1
    config["evaluation_duration"] = 20
    config["evaluation_duration_unit"] = "episodes"
    config["metrics_num_episodes_for_smoothing"] = 20

    config["multiagent"] = {"policies": policies, "policy_mapping_fn": (lambda agent_id, episode, **kwargs: agent_id)}

    if args.battle_advpursuit_against_pretrained is True:
        ModelCatalog.register_custom_model("BattleModelUnflat", BattleModelUnflat)
        # config["callbacks"] = BattleCallback
        config["multiagent"]["policies_to_train"] = [pol for pol in config["multiagent"]["policies"] if pol[:4] == "blue"]
        # model_config = {
        #     "model": {
        #         "custom_model": "BattleModelUnflat",
        #     },
        #     "gamma": 0.99,
        # }
        # for pol in config["multiagent"]["policies"]:
        #     if pol[:3] == "red":
        #         config["multiagent"]["policies"][pol] = PolicySpec(my_custom_dqn_unflat, None, None, model_config)
        # config["simple_optimizer"] = True
        # The following option effecrtively removes the red agents from the env, as far as we on the outside can see.
        # This means that the above with creating policies for them is pointless, but I'll leave this for now just in case.
        config["env_config"]["wrap_pretrained_agents"] = True
        config["multiagent"]["policies"] = {pol: config["multiagent"]["policies"][pol] for pol in config["multiagent"]["policies"] if pol[:4] == "blue"}

    stop = {"timesteps_total": _BATTLE_N_TIMESTEPS}

    return MADDPG, config, stop


def config_battlev4_qmix(args):

    env_config = {"map_size": 18, "group_agents": True}

    # Overwrite env_config variables using CLI args, if given.
    if args.env_battle_map_size is not None:
        env_config["map_size"] = args.env_battle_map_size

    env = battle_v4.parallel_env(map_size=env_config["map_size"], minimap_mode=False, step_reward=-0.005, dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2, max_cycles=1000, extra_features=False)

    num_battle_agents = len(env.agents)

    policies = {"group_1": PolicySpec(), "group_2": PolicySpec()}

    config = deepcopy(QMixConfig().to_dict())  # deepcopy(get_trainer_class(alg_name).get_default_config())

    config["env"] = "battle"

    config["env_config"] = env_config
    config["simple_optimizer"] = True
    config["num_gpus"] = 0
    config["framework"] = "torch"
    # config["log_level"] = "DEBUG"
    # config['replay_buffer_config']['capacity'] = 120000
    config["exploration_config"] = {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 0.1,
        "final_epsilon": 0.001,
        "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
    }
    config["replay_buffer_config"]["type"] = "MultiAgentPrioritizedReplayBuffer"
    config["replay_buffer_config"]["prioritized_replay_alpha"] = 0.6
    config["replay_buffer_config"]["prioritized_replay_beta"] = 0.4
    config["replay_buffer_config"]["prioritized_replay_eps"] = 1e-06

    config["horizon"] = 1000
    # config["dueling"] = True
    # config["target_network_update_freq"] = 1000
    config["rollout_fragment_length"] = 4
    config["no_done_at_end"] = False

    if args.parameter_tunning is not None:
        config["lr"] = tune.grid_search([0.0001, 0.0002, 0.00025, 0.0003])
        config["train_batch_size"] = tune.grid_search([32, 64, 128, 256])
    else:
        config["lr"] = 0.00016
        config["train_batch_size"] = 32

    config["min_sample_timesteps_per_iteration"] = _BATTLE_N_TIMESTEPS / _EVAL_RESOLUTION
    config["min_time_s_per_iteration"] = 0
    config["evaluation_interval"] = 1
    config["evaluation_duration"] = 20
    config["evaluation_duration_unit"] = "episodes"
    config["metrics_num_episodes_for_smoothing"] = 20

    # config["num_steps_sampled_before_learning_starts"] = 1000
    config["multiagent"] = {"policies": policies, "policy_mapping_fn": (lambda agent_id, episode, **kwargs: agent_id)}

    # if fully connected is true, use fully connected NN instead of conv based network.
    if args.fully_connected == "true":
        config["model"] = {"fcnet_hiddens": [64, 64]}
        config["env_config"]["flatten_obs"] = True

    if args.battle_advpursuit_against_pretrained is True:
        ModelCatalog.register_custom_model("BattleModel", BattleModel)
        config["callbacks"] = BattleCallback
        # config["multiagent"]["policies_to_train"] = [pol for pol in config["multiagent"]["policies"] if pol[:4] == "blue"]
        config["multiagent"]["policies_to_train"] = ["group_1"]
        model_config = {
            "model": {
                "custom_model": "BattleModel",
            },
            "gamma": 0.99,
        }
        config["multiagent"]["policies"] = {
            "red_{}".format(i): PolicySpec(my_custom_dqn, spaces.Box(0.0, 2.0, (845,) if args.fully_connected == "true" else (13, 13, 5), np.float32), spaces.Discrete(21), model_config) for i in range(int(num_battle_agents / 2))
        }
        config["multiagent"]["policies"]["group_1"] = PolicySpec(
            None, spaces.Tuple([spaces.Box(0.0, 2.0, (845,) if args.fully_connected == "true" else (13, 13, 5), np.float32) for i in range(int(num_battle_agents / 2))]), spaces.Tuple([spaces.Discrete(21) for i in range(int(num_battle_agents / 2))]), None
        )
        config["simple_optimizer"] = True
        config["env_config"]["group_agents"] = "blueonly"

    stop = {"timesteps_total": _BATTLE_N_TIMESTEPS}

    return QMix, config, stop


"""
##################################
Adversarial pursuit config functions
for DQN,QMIX and MADDPG.
##################################
"""


class AdvPCallback(DefaultCallbacks):
    def on_algorithm_init(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,
    ) -> None:
        """Loads weights from a checkpoint, and sets the algorithm's weights to the loaded weights for red team only."""
        checkpoint_file = os.path.dirname(os.path.realpath(__file__)) + "/checkpoints/advpursuit.weights"
        with open(checkpoint_file, "rb") as f:
            weights = pickle.load(f)
        algorithm.set_weights(weights)


def my_custom_dqn_advp(
    observation_space,
    action_space,
    config,
):
    config["hiddens"] = [256]
    config["dueling"] = True
    config["double_q"] = True
    config["model"] = {
        "_use_default_native_models": False,
        "_disable_preprocessor_api": False,
        "_disable_action_flattening": False,
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
        "conv_filters": None,
        "conv_activation": "relu",
        "post_fcnet_hiddens": [],
        "post_fcnet_activation": "relu",
        "free_log_std": False,
        "no_final_linear": False,
        "vf_share_layers": True,
        "use_lstm": False,
        "max_seq_len": 20,
        "lstm_cell_size": 256,
        "lstm_use_prev_action": False,
        "lstm_use_prev_reward": False,
        "_time_major": False,
        "use_attention": False,
        "attention_num_transformer_units": 1,
        "attention_dim": 64,
        "attention_num_heads": 1,
        "attention_head_dim": 32,
        "attention_memory_inference": 50,
        "attention_memory_training": 50,
        "attention_position_wise_mlp_dim": 32,
        "attention_init_gru_gate_bias": 2.0,
        "attention_use_n_prev_actions": 0,
        "attention_use_n_prev_rewards": 0,
        "framestack": True,
        "dim": 84,
        "grayscale": False,
        "zero_mean": True,
        "custom_model": "AdversarialPursuitModel",
        "custom_model_config": {},
        "custom_action_dist": None,
        "custom_preprocessor": None,
        "lstm_use_prev_action_reward": -1,
    }
    return DQNTorchPolicy(observation_space, action_space, config)


def my_custom_dqn_unflat_advp(
    observation_space,
    action_space,
    config,
):
    config["hiddens"] = [256]
    config["dueling"] = True
    config["double_q"] = True
    config["model"] = {
        "_use_default_native_models": False,
        "_disable_preprocessor_api": False,
        "_disable_action_flattening": False,
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
        "conv_filters": None,
        "conv_activation": "relu",
        "post_fcnet_hiddens": [],
        "post_fcnet_activation": "relu",
        "free_log_std": False,
        "no_final_linear": False,
        "vf_share_layers": True,
        "use_lstm": False,
        "max_seq_len": 20,
        "lstm_cell_size": 256,
        "lstm_use_prev_action": False,
        "lstm_use_prev_reward": False,
        "_time_major": False,
        "use_attention": False,
        "attention_num_transformer_units": 1,
        "attention_dim": 64,
        "attention_num_heads": 1,
        "attention_head_dim": 32,
        "attention_memory_inference": 50,
        "attention_memory_training": 50,
        "attention_position_wise_mlp_dim": 32,
        "attention_init_gru_gate_bias": 2.0,
        "attention_use_n_prev_actions": 0,
        "attention_use_n_prev_rewards": 0,
        "framestack": True,
        "dim": 84,
        "grayscale": False,
        "zero_mean": True,
        "custom_model": "AdversarialPursuitModelUnflat",
        "custom_model_config": {},
        "custom_action_dist": None,
        "custom_preprocessor": None,
        "lstm_use_prev_action_reward": -1,
    }
    return DQNTorchPolicy(observation_space, action_space, config)


def policy_mapping_fn_adv_pursuit(agent_id, episode, worker, **kwargs):
    team = agent_id.split("_")[0]
    if team == "prey":  # blue team is using parameter sharing
        return "prey_"
    else:
        return agent_id  # red team is not using parameter sharing


def config_adversarial_pursuit_dqn(args):
    # num of predators and prey is determined according to map size,
    # and is not given as input to the environment.
    num_predators = 4
    num_preys = 8

    # map size sets dimensions of the (square) map.
    env_config = {"map_size": 18}

    # Overwrite env_config variables using CLI args, if given.
    if args.env_adversarial_pursuit_map_size is not None:
        env_config["map_size"] = args.env_adversarial_pursuit_map_size

    ModelCatalog.register_custom_model("AdversarialPursuitModel", AdversarialPursuitModel)

    # if fully connected is true, use fully connected NN instead of conv based network.
    if args.fully_connected == "true":
        model_config = None
    else:
        model_config = {
            "model": {
                "custom_model": "AdversarialPursuitModel",
            },
            "gamma": 0.99,
        }

    policies_predator_battle = ["predator_{}".format(i) for i in range(num_predators)]
    policies_prey_battle = ["prey_{}".format(i) for i in range(num_preys)]
    policies_list = policies_prey_battle + policies_predator_battle

    if args.parameter_sharing == "true":
        # if parameter_sharing=True, only one policy is used for all agents.
        policies = {"predator_{}".format(i): PolicySpec(None, None, None, model_config) for i in range(num_predators)}
        policies_prey = {"prey_": PolicySpec(None, None, None, model_config)}
        policies.update(policies_prey)
    else:
        policies = {i: PolicySpec(None, None, None, model_config) for i in policies_list}

    config = deepcopy(suPERDQNConfig().to_dict())  # deepcopy(get_trainer_class(alg_name).get_default_config())
    config["env"] = "adversarial_pursuit"
    config["env_config"] = env_config
    config["store_buffer_in_checkpoints"] = False
    config["num_gpus"] = 0
    config["framework"] = "torch"
    # config["log_level"] = "DEBUG"
    config["replay_buffer_config"]["capacity"] = 90000
    config["exploration_config"] = {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 0.1,
        "final_epsilon": 0.001,
        "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
    }
    config["replay_buffer_config"]["type"] = "MultiAgentPrioritizedReplayBuffer"
    config["rollout_fragment_length"] = 5
    config["replay_buffer_config"]["prioritized_replay_alpha"] = 0.6
    config["replay_buffer_config"]["prioritized_replay_beta"] = 0.4
    config["replay_buffer_config"]["prioritized_replay_eps"] = 0.00001
    config["train_batch_size"] = (num_predators + num_preys) * 32  # each agent's sample batch is with size 32
    config["replay_buffer_config"]["worker_side_prioritization"] = True
    config["lr"] = 1e-4
    config["horizon"] = 800
    config["dueling"] = True
    config["target_network_update_freq"] = 1200
    config["no_done_at_end"] = False
    config["multiagent"] = {"policies": policies}

    if args.parameter_sharing == "true":
        # if parameter_sharing=True, map all agents from the same team into a single policy.
        config["multiagent"]["policy_mapping_fn"] = policy_mapping_fn_adv_pursuit
    else:
        config["multiagent"]["policy_mapping_fn"] = lambda agent_id, episode, **kwargs: agent_id

    config["min_sample_timesteps_per_iteration"] = _ADVPURSUIT_N_TIMESTEPS / _EVAL_RESOLUTION
    config["min_time_s_per_iteration"] = 0
    config["evaluation_interval"] = 1
    config["evaluation_duration"] = 20
    config["evaluation_duration_unit"] = "episodes"
    config["metrics_num_episodes_for_smoothing"] = 20

    config["num_steps_sampled_before_learning_starts"] = 1000

    # if fully connected is true, use fully connected NN instead of conv based network.
    if args.fully_connected == "true":
        config["model"] = {"fcnet_hiddens": [64, 64]}
        config["env_config"]["flatten_obs"] = True

    if args.battle_advpursuit_against_pretrained is True:
        ModelCatalog.register_custom_model("AdversarialPursuitModel", AdversarialPursuitModel)
        config["callbacks"] = AdvPCallback
        config["multiagent"]["policies_to_train"] = [pol for pol in config["multiagent"]["policies"] if pol[:4] == "prey"]
        for pol in config["multiagent"]["policies"]:
            if pol[:5] == "preda":
                config["multiagent"]["policies"][pol] = PolicySpec(my_custom_dqn_advp, spaces.Box(0.0, 2.0, (500,) if args.fully_connected == "true" else (10, 10, 5), np.float32), spaces.Discrete(13), model_config)
        config["simple_optimizer"] = False

    stop = {"timesteps_total": _ADVPURSUIT_N_TIMESTEPS}

    return suPERDQN, config, stop


def config_adversarial_pursuit_plaindqn(args):
    # num of predators and prey is determined according to map size,
    # and is not given as input to the environment.
    num_predators = 4
    num_preys = 8

    # map size sets dimensions of the (square) map.
    env_config = {"map_size": 18}

    # Overwrite env_config variables using CLI args, if given.
    if args.env_adversarial_pursuit_map_size is not None:
        env_config["map_size"] = args.env_adversarial_pursuit_map_size

    ModelCatalog.register_custom_model("AdversarialPursuitModel", AdversarialPursuitModel)

    # if fully connected is true, use fully connected NN instead of conv based network.
    if args.fully_connected == "true":
        model_config = None
    else:
        model_config = {
            "model": {
                "custom_model": "AdversarialPursuitModel",
            },
            "gamma": 0.99,
        }

    policies_predator_battle = ["predator_{}".format(i) for i in range(num_predators)]
    policies_prey_battle = ["prey_{}".format(i) for i in range(num_preys)]
    policies_list = policies_prey_battle + policies_predator_battle

    if args.parameter_sharing == "true":
        # if parameter_sharing=True, only one policy is used for all agents.
        policies = {"predator_{}".format(i): PolicySpec(None, None, None, model_config) for i in range(num_predators)}
        policies_prey = {"prey_": PolicySpec(None, None, None, model_config)}
        policies.update(policies_prey)
    else:
        policies = {i: PolicySpec(None, None, None, model_config) for i in policies_list}

    config = deepcopy(suPERDQNConfig().to_dict())  # deepcopy(get_trainer_class(alg_name).get_default_config())
    config["env"] = "adversarial_pursuit"
    config["env_config"] = env_config
    config["store_buffer_in_checkpoints"] = False
    config["num_gpus"] = 0
    config["framework"] = "torch"
    # config["log_level"] = "DEBUG"
    config["replay_buffer_config"]["capacity"] = 90000
    config["exploration_config"] = {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 0.1,
        "final_epsilon": 0.001,
        "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
    }
    config["replay_buffer_config"]["type"] = "MultiAgentPrioritizedReplayBuffer"
    config["rollout_fragment_length"] = 5
    config["replay_buffer_config"]["prioritized_replay_alpha"] = 0.6
    config["replay_buffer_config"]["prioritized_replay_beta"] = 0.4
    config["replay_buffer_config"]["prioritized_replay_eps"] = 0.00001
    config["train_batch_size"] = (num_predators + num_preys) * 32  # each agent's sample batch is with size 32
    config["replay_buffer_config"]["worker_side_prioritization"] = True
    config["lr"] = 1e-4
    config["horizon"] = 800
    config["dueling"] = False
    config["double_q"] = False
    config["target_network_update_freq"] = 1200
    config["no_done_at_end"] = False
    config["multiagent"] = {"policies": policies}

    if args.parameter_sharing == "true":
        # if parameter_sharing=True, map all agents from the same team into a single policy.
        config["multiagent"]["policy_mapping_fn"] = policy_mapping_fn_adv_pursuit
    else:
        config["multiagent"]["policy_mapping_fn"] = lambda agent_id, episode, **kwargs: agent_id

    config["min_sample_timesteps_per_iteration"] = _ADVPURSUIT_N_TIMESTEPS / _EVAL_RESOLUTION
    config["min_time_s_per_iteration"] = 0
    config["evaluation_interval"] = 1
    config["evaluation_duration"] = 20
    config["evaluation_duration_unit"] = "episodes"
    config["metrics_num_episodes_for_smoothing"] = 20

    config["num_steps_sampled_before_learning_starts"] = 1000

    # if fully connected is true, use fully connected NN instead of conv based network.
    if args.fully_connected == "true":
        config["model"] = {"fcnet_hiddens": [64, 64]}
        config["env_config"]["flatten_obs"] = True

    if args.battle_advpursuit_against_pretrained is True:
        ModelCatalog.register_custom_model("AdversarialPursuitModel", AdversarialPursuitModel)
        config["callbacks"] = AdvPCallback
        config["multiagent"]["policies_to_train"] = [pol for pol in config["multiagent"]["policies"] if pol[:4] == "prey"]
        # config["multiagent"]["policies_to_train"] = ["group_1"]
        # model_config = {
        #     "model": {
        #         "custom_model": "BattleModel",
        #     },
        #     "gamma": 0.99,
        # }
        for pol in config["multiagent"]["policies"]:
            if pol[:5] == "preda":
                config["multiagent"]["policies"][pol] = PolicySpec(my_custom_dqn_advp, spaces.Box(0.0, 2.0, (500,) if args.fully_connected == "true" else (10, 10, 5), np.float32), spaces.Discrete(13), model_config)
        config["simple_optimizer"] = False

    stop = {"timesteps_total": _ADVPURSUIT_N_TIMESTEPS}

    return suPERDQN, config, stop


def config_adversarial_pursuit_maddpg(args):
    # Only do MADDPG against pretrained, this is hardcoded in env now.
    assert args.battle_advpursuit_against_pretrained is True
    env_config = {"map_size": 18, "flatten_obs": True, "actions_are_logits": True}
    num_predators = 4
    num_preys = 8
    # Overwrite env_config variables using CLI args, if given.
    if args.env_battle_map_size is not None:
        env_config["map_size"] = args.env_adversarial_pursuit_map_size

    policies_predator_battle = ["predator_{}".format(i) for i in range(num_predators)]
    policies_prey_battle = ["prey_{}".format(i) for i in range(num_preys)]
    policies_list = policies_prey_battle + policies_predator_battle
    policies = {i: PolicySpec(None, None, None, config={"agent_id": count}) for count, i in enumerate(policies_list)}
    config = deepcopy(MADDPGConfig().to_dict())  # deepcopy(get_trainer_class(alg_name).get_default_config())

    config["env"] = "adversarial_pursuit"

    config["env_config"] = env_config

    config["num_gpus"] = 0
    config["framework"] = "tf"
    config["exploration_config"] = {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 0.1,
        "final_epsilon": 0.001,
        "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
    }

    config["replay_buffer_config"]["capacity"] = 90000
    config["replay_buffer_config"]["learning_starts"] = 0

    config["horizon"] = 800
    config["rollout_fragment_length"] = 32
    config["no_done_at_end"] = False

    config["critic_hiddens"] = [64, 64]
    config["actor_hiddens"] = [64, 64]

    # if set to True, training is performed with gridserch (multiple experiments)
    if args.parameter_tunning is not None:
        config["actor_feature_reg"] = tune.grid_search([0.001, 0.0015, 0.002])
        config["critic_lr"] = config["actor_lr"] = tune.grid_search([0.0001, 0.0002, 0.00025, 0.0003])
        config["train_batch_size"] = tune.grid_search([32, 64, 128, 256])
        config["tau"] = tune.grid_search([0.05, 0.01, 0.015, 0.2])
    else:  # standard training
        config["actor_feature_reg"] = 0.001
        config["critic_lr"] = config["actor_lr"] = 0.00025
        config["train_batch_size"] = 32
        config["tau"] = 0.015

    config["min_sample_timesteps_per_iteration"] = _ADVPURSUIT_N_TIMESTEPS / _EVAL_RESOLUTION
    config["min_time_s_per_iteration"] = 0
    config["evaluation_interval"] = 1
    config["evaluation_duration"] = 20
    config["evaluation_duration_unit"] = "episodes"
    config["metrics_num_episodes_for_smoothing"] = 20

    config["multiagent"] = {"policies": policies, "policy_mapping_fn": (lambda agent_id, episode, **kwargs: agent_id)}

    if args.battle_advpursuit_against_pretrained is True:
        ModelCatalog.register_custom_model("AdversarialPursuitModelUnflat", AdversarialPursuitModelUnflat)
        # config["callbacks"] = AdvPCallback
        config["multiagent"]["policies_to_train"] = [pol for pol in config["multiagent"]["policies"] if pol[:4] == "prey"]
        # model_config = {
        #     "model": {
        #         "custom_model": "AdversarialPursuitModelUnflat",
        #     },
        #     "gamma": 0.99,
        # }
        # for pol in config["multiagent"]["policies"]:
        #     if pol[:5] == "preda":
        #         config["multiagent"]["policies"][pol] = PolicySpec(my_custom_dqn_unflat_advp, None, None, model_config)
        # config["simple_optimizer"] = False
        config["env_config"]["wrap_pretrained_agents"] = True
        config["multiagent"]["policies"] = {pol: config["multiagent"]["policies"][pol] for pol in config["multiagent"]["policies"] if pol[:4] == "prey"}

    stop = {"timesteps_total": _ADVPURSUIT_N_TIMESTEPS}

    return MADDPG, config, stop


def config_adversarial_pursuit_qmix(args):
    env_config = {"map_size": 18, "group_agents": True}
    num_predators = 4
    num_preys = 8
    # Overwrite env_config variables using CLI args, if given.
    if args.env_battle_map_size is not None:
        env_config["map_size"] = args.env_adversarial_pursuit_map_size

    env = env_creator_adversarial_pursuit({"map_size": env_config["map_size"], "group_agents": False})

    grouping = {
        "group_2": ["predator_{}".format(i) for i in range(num_predators)],
        "group_1": ["prey_{}".format(i) for i in range(num_preys)],
    }
    obs_space_prey = spaces.Tuple([env.observation_space for agent in grouping["group_1"]])
    obs_space_predator = spaces.Tuple([env.observation_space for agent in grouping["group_2"]])

    act_space_prey = spaces.Tuple([env.action_space for agent in grouping["group_1"]])
    act_space_predator = spaces.Tuple([env.action_space for agent in grouping["group_2"]])

    policies = {"group_2": PolicySpec(observation_space=obs_space_predator, action_space=act_space_predator), "group_1": PolicySpec(observation_space=obs_space_prey, action_space=act_space_prey)}

    config = deepcopy(QMixConfig().to_dict())  # deepcopy(get_trainer_class(alg_name).get_default_config())

    config["env"] = "adversarial_pursuit"

    config["env_config"] = env_config
    config["simple_optimizer"] = True
    config["num_gpus"] = 0
    config["framework"] = "torch"
    # config["log_level"] = "DEBUG"
    # config['replay_buffer_config']['capacity'] = 120000
    config["exploration_config"] = {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 0.1,
        "final_epsilon": 0.001,
        "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
    }
    config["replay_buffer_config"]["type"] = "MultiAgentPrioritizedReplayBuffer"
    config["replay_buffer_config"]["prioritized_replay_alpha"] = 0.6
    config["replay_buffer_config"]["prioritized_replay_beta"] = 0.4
    config["replay_buffer_config"]["prioritized_replay_eps"] = 1e-06
    config["horizon"] = 800
    # config["dueling"] = True
    # config["target_network_update_freq"] = 1000
    config["rollout_fragment_length"] = 4
    config["no_done_at_end"] = False

    if args.parameter_tunning is not None:
        config["lr"] = tune.grid_search([0.0001, 0.0002, 0.00025, 0.0003])
        config["train_batch_size"] = tune.grid_search([32, 64, 128, 256])
    else:
        config["lr"] = 0.00016
        config["train_batch_size"] = 32

    config["min_sample_timesteps_per_iteration"] = _ADVPURSUIT_N_TIMESTEPS / _EVAL_RESOLUTION
    config["min_time_s_per_iteration"] = 0
    config["evaluation_interval"] = 1
    config["evaluation_duration"] = 20
    config["evaluation_duration_unit"] = "episodes"
    config["metrics_num_episodes_for_smoothing"] = 20

    # config["num_steps_sampled_before_learning_starts"] = 1000
    config["multiagent"] = {"policies": policies, "policy_mapping_fn": (lambda agent_id, episode, **kwargs: agent_id)}

    # if fully connected is true, use fully connected NN instead of conv based network.
    if args.fully_connected == "true":
        config["model"] = {"fcnet_hiddens": [64, 64]}
        config["env_config"]["flatten_obs"] = True

    if args.battle_advpursuit_against_pretrained is True:
        ModelCatalog.register_custom_model("AdversarialPursuitModel", AdversarialPursuitModel)
        config["callbacks"] = AdvPCallback
        # config["multiagent"]["policies_to_train"] = [pol for pol in config["multiagent"]["policies"] if pol[:4] == "prey"]
        config["multiagent"]["policies_to_train"] = ["group_1"]
        model_config = {
            "model": {
                "custom_model": "AdversarialPursuitModel",
            },
            "gamma": 0.99,
        }
        config["multiagent"]["policies"] = {
            "predator_{}".format(i): PolicySpec(my_custom_dqn_advp, spaces.Box(0.0, 2.0, (500,) if args.fully_connected == "true" else (10, 10, 5), np.float32), spaces.Discrete(13), model_config) for i in range(num_predators)
        }
        config["multiagent"]["policies"]["group_1"] = PolicySpec(
            None, spaces.Tuple([spaces.Box(0.0, 2.0, (500,) if args.fully_connected == "true" else (10, 10, 5), np.float32) for i in range(num_preys)]), spaces.Tuple([spaces.Discrete(13) for i in range(num_preys)]), None
        )
        config["simple_optimizer"] = True
        config["env_config"]["group_agents"] = "preyonly"

    #     for pol in config["multiagent"]["policies"]:
    #         if pol[:5] == "preda":
    #             config["multiagent"]["policies"][pol] = PolicySpec(my_custom_dqn_advp, spaces.Box(0.0, 2.0, (10, 10, 5), np.float32), spaces.Discrete(13), model_config)
    #     config["simple_optimizer"] = False

    # if args.battle_advpursuit_against_pretrained is True:
    #     ModelCatalog.register_custom_model("BattleModel", BattleModel)
    #     config["callbacks"] = BattleCallback
    #     # config["multiagent"]["policies_to_train"] = [pol for pol in config["multiagent"]["policies"] if pol[:4] == "blue"]
    #     config["multiagent"]["policies_to_train"] = ["group_1"]
    #     model_config = {
    #         "model": {
    #             "custom_model": "BattleModel",
    #         },
    #         "gamma": 0.99,
    #     }
    #     config["multiagent"]["policies"] = {"red_{}".format(i): PolicySpec(my_custom_dqn, spaces.Box(0.0, 2.0, (13, 13, 5), np.float32), spaces.Discrete(21), model_config) for i in range(int(num_battle_agents / 2))}
    #     config["multiagent"]["policies"]["group_1"] = PolicySpec(
    #         None, spaces.Tuple([spaces.Box(0.0, 2.0, (13, 13, 5), np.float32) for i in range(int(num_battle_agents / 2))]), spaces.Tuple([spaces.Discrete(21) for i in range(int(num_battle_agents / 2))]), None
    #     )
    #     config["simple_optimizer"] = True
    #     config["env_config"]["group_agents"] = "blueonly"

    stop = {"timesteps_total": _ADVPURSUIT_N_TIMESTEPS}

    return QMix, config, stop
