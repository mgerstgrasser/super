import os
import pickle

import numpy as np
import ray
import supersuit
from gym import spaces
from pettingzoo.magent import adversarial_pursuit_v4, battle_v4
from pettingzoo.sisl import pursuit_v4
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from supersuit import black_death_v3, pad_action_space_v0, pad_observations_v0


class manual_wrap(ParallelPettingZooEnv):
    """This class is used only for the Pursuit environment. It inherits
    the ParallelPettingZooEnv wrapper, and overwrites the reset logic. In the current
    version of Pursuit, seeding the environment cannot be done via the config file, but rather
    through the reset command. RLlib don't support seeding via reset yet, hence
    we provide this customized wrapper class.
    """

    def __init__(self, env, seed, actions_are_logits):
        super().__init__(env)
        env.reset(seed=seed)
        self.actions_are_logits = actions_are_logits

    def reset(self):
        return self.par_env.reset()

    def step(self, action_dict):
        # MADDPG emits action logits instead of actual discrete actions
        if self.actions_are_logits is True:
            for agent in action_dict:
                # Ugly hardcoding this to only do this for agent we train in MADDPG experiments
                if agent[:4] == "prey" or agent[:4] == "blue" or agent[:4] == "purs":
                    action_dict[agent] = np.random.choice([i for i in range(len(action_dict[agent]))], p=action_dict[agent])
            # action_dict = {k: np.random.choice([i for i in range(len(v))], p=v) for k, v in action_dict.items()}

        obss, rews, dones, infos = super().step(action_dict)
        return obss, rews, dones, infos

    def seed(self, seed=None):
        pass


def env_creator_pursuit(env_config):
    """This function creates a parallel Pursuit environment object.
    We use default configurations provided by petting zoo.
    """
    env = pursuit_v4.parallel_env(
        max_cycles=500,
        x_size=16,
        y_size=16,
        shared_reward=env_config["shared_reward"],
        n_evaders=env_config["n_evaders"],
        n_pursuers=env_config["num_agents"],
        obs_range=7,
        n_catch=2,
        freeze_evaders=False,
        tag_reward=0.01,
        catch_reward=5.0,
        urgency_reward=-0.1,
        surround=True,
        constraint_window=1.0,
    )
    if env_config.get("normalize_obs", False):
        env = supersuit.dtype_v0(env, np.float32)
        env = supersuit.normalize_obs_v0(env)

    #  when using MADDPG observation space is flattened.
    if env_config.get("flatten_obs", False):
        env = supersuit.flatten_v0(env)

    env = manual_wrap(env, env_config.get("seed", None), env_config.get("actions_are_logits", False))
    if env_config.get("group_agents", False):
        agents = env.par_env.agents
        grouping = {
            "group_1": ["pursuer_{}".format(i) for i in range(env_config["num_agents"])],
        }
        obs_space = spaces.Tuple([env.par_env.observation_space(agent) for agent in agents])
        act_space = spaces.Tuple([env.par_env.action_space(agent) for agent in agents])
        env = env.with_agent_groups(grouping, obs_space=obs_space, act_space=act_space)
    return env


def env_creator_battle(env_config):
    """This function creates a parallel Battled environment object.
    We use default configurations provided by petting zoo.
    """
    env = battle_v4.parallel_env(map_size=env_config["map_size"], minimap_mode=False, step_reward=-0.005, dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2, max_cycles=1000, extra_features=False)

    #  when using MADDPG observation space is flattened.
    if env_config.get("flatten_obs", False):
        env = supersuit.flatten_v0(env)
    if env_config.get("group_agents", False) or env_config.get("black_death", False) or env_config.get("wrap_pretrained_agents", False):
        env = black_death_v3(env)
    if env_config.get("wrap_pretrained_agents", False):
        env = manual_wrap_battle_pretrained_agents(env, env_config.get("seed", None), env_config.get("actions_are_logits", False))
    else:
        env = manual_wrap(env, env_config.get("seed", None), env_config.get("actions_are_logits", False))
    if env_config.get("group_agents", False):
        agents = env.par_env.agents
        num_battle_agents = len(agents)

        if env_config.get("group_agents", False) == "blueonly":
            grouping = {"group_1": ["blue_{}".format(i) for i in range(int(num_battle_agents / 2))]}
            env = env.with_agent_groups(grouping)
        else:
            grouping = {
                "group_1": ["blue_{}".format(i) for i in range(int(num_battle_agents / 2))],
                "group_2": ["red_{}".format(i) for i in range(int(num_battle_agents / 2))],
            }
            obs_space = spaces.Tuple([env.observation_space for agent in grouping["group_1"]])
            act_space = spaces.Tuple([env.action_space for agent in grouping["group_1"]])
            env = env.with_agent_groups(grouping, obs_space=obs_space, act_space=act_space)
    return env


def env_creator_adversarial_pursuit(env_config):
    env = adversarial_pursuit_v4.parallel_env(map_size=env_config["map_size"], minimap_mode=False, tag_penalty=-0.2, max_cycles=500, extra_features=False)

    # Predator and prey have different action space and state sizes,
    # since Rlib requires all agents to have identical state and action size,
    # padding is required.
    env = pad_observations_v0(env)
    env = pad_action_space_v0(env)

    #  when using MADDPG observation space is flattened.
    if env_config.get("flatten_obs", False):
        env = supersuit.flatten_v0(env)

    if env_config.get("wrap_pretrained_agents", False):
        env = manual_wrap_advp_pretrained_agents(env, env_config.get("seed", None), env_config.get("actions_are_logits", False))
    else:
        env = manual_wrap(env, env_config.get("seed", None), env_config.get("actions_are_logits", False))

    if env_config.get("group_agents", False):
        num_predators = 4
        num_preys = 8

        if env_config.get("group_agents", False) == "preyonly":
            grouping = {"group_1": ["prey_{}".format(i) for i in range(num_preys)]}
            env = env.with_agent_groups(grouping)
        else:
            grouping = {
                "group_1": ["prey_{}".format(i) for i in range(num_preys)],
                "group_2": ["predator_{}".format(i) for i in range(num_predators)],
            }
            env = env.with_agent_groups(grouping)

    return env


class manual_wrap_battle_pretrained_agents(ParallelPettingZooEnv):
    """This is a super hacky way to quickly get pre-trained agents inside the environment."""

    def __init__(self, env, seed, actions_are_logits):
        super().__init__(env)
        env.reset(seed=seed)
        self.actions_are_logits = actions_are_logits

        from suPER.experiments.main import get_trainer_and_config, parse_args

        args, _ = parse_args("--suPER=False --suPER_bandwidth=0.0 --battle_advpursuit_against_pretrained --env=battle --use_gpu=0".split())
        trainer, config, stop = get_trainer_and_config(args)

        if not ray.is_initialized():
            ray.init(num_cpus=1, local_mode=True)

        self.trainer = trainer(config=config)
        # Probably not needed, but just in case:
        checkpoint_file = os.path.dirname(os.path.realpath(__file__)) + "/checkpoints/battle.weights"
        with open(checkpoint_file, "rb") as f:
            weights = pickle.load(f)
        self.trainer.set_weights(weights)
        pass

        self.par_env.agents = [agent for agent in self.par_env.agents if agent[:4] == "blue"]

    def reset(self):
        obs = self.par_env.reset()
        self.obs = obs
        return {agent: obs[agent] for agent in obs if agent[:4] == "blue"}

    def step(self, action_dict):
        # MADDPG emits action logits instead of actual discrete actions
        if self.actions_are_logits is True:
            for agent in action_dict:
                # Ugly hardcoding this to only do this for agent we train in MADDPG experiments
                if agent[:4] == "prey" or agent[:4] == "blue" or agent[:4] == "purs":
                    action_dict[agent] = np.random.choice([i for i in range(len(action_dict[agent]))], p=action_dict[agent])
            # action_dict = {k: np.random.choice([i for i in range(len(v))], p=v) for k, v in action_dict.items()}

        # Get actions from pre-trained agents
        for agent in [f"red_{i}" for i in range(6)]:
            action_dict[agent] = self.trainer.compute_action(self.obs[agent], policy_id=agent)

        obs, rews, dones, infos = super().step(action_dict)
        self.obs = obs
        return {agent: obs[agent] for agent in obs if agent[:4] == "blue"}, {agent: rews[agent] for agent in rews if agent[:4] == "blue"}, {agent: dones[agent] for agent in dones if agent[:3] != "red"}, {}

    def seed(self, seed=None):
        pass


class manual_wrap_advp_pretrained_agents(ParallelPettingZooEnv):
    """This is a super hacky way to quickly get pre-trained agents inside the environment."""

    def __init__(self, env, seed, actions_are_logits):
        super().__init__(env)
        env.reset(seed=seed)
        self.actions_are_logits = actions_are_logits

        from suPER.experiments.main import get_trainer_and_config, parse_args

        args, _ = parse_args("--suPER=False --suPER_bandwidth=0.0 --battle_advpursuit_against_pretrained --env=adversarial_pursuit --use_gpu=0".split())
        trainer, config, stop = get_trainer_and_config(args)

        if not ray.is_initialized():
            ray.init(num_cpus=1, local_mode=True)

        self.trainer = trainer(config=config)
        # Probably not needed, but just in case:
        checkpoint_file = os.path.dirname(os.path.realpath(__file__)) + "/checkpoints/advpursuit.weights"
        with open(checkpoint_file, "rb") as f:
            weights = pickle.load(f)
        self.trainer.set_weights(weights)
        pass

        self.par_env.agents = [agent for agent in self.par_env.agents if agent[:4] == "prey"]

    def reset(self):
        obs = self.par_env.reset()
        self.obs = obs
        return {agent: obs[agent] for agent in obs if agent[:4] == "prey"}

    def step(self, action_dict):
        # MADDPG emits action logits instead of actual discrete actions
        if self.actions_are_logits is True:
            for agent in action_dict:
                # Ugly hardcoding this to only do this for agent we train in MADDPG experiments
                if agent[:4] == "prey" or agent[:4] == "blue" or agent[:4] == "purs":
                    action_dict[agent] = np.random.choice([i for i in range(len(action_dict[agent]))], p=action_dict[agent])
            # action_dict = {k: np.random.choice([i for i in range(len(v))], p=v) for k, v in action_dict.items()}

        # Get actions from pre-trained agents
        for agent in [f"predator_{i}" for i in range(4)]:
            action_dict[agent] = self.trainer.compute_action(self.obs[agent], policy_id=agent)

        obs, rews, dones, infos = super().step(action_dict)
        self.obs = obs
        return {agent: obs[agent] for agent in obs if agent[:4] == "prey"}, {agent: rews[agent] for agent in rews if agent[:4] == "prey"}, {agent: dones[agent] for agent in dones if agent[:4] != "pred"}, {}

    def seed(self, seed=None):
        pass
