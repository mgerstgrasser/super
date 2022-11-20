"""Simple example of setting up a multi-agent policy mapping.

Control the number of agents and policies via --num-agents and --num-policies.

This works with hundreds of agents and policies, but note that initializing
many TF policies will take some time.

Also, TF evals might slow down with large numbers of policies. To debug TF
execution, set the TF_TIMELINE_DIR environment variable.
"""
import os
import random

import numpy as np
import torch
from gym import spaces
from gym.utils import seeding
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.misc import SlimConv2d, SlimFC
from ray.rllib.models.torch.misc import normc_initializer as normc_initializer_torch
from ray.rllib.models.torch.misc import same_padding
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn


class PursuitModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, [2, 2]),
            nn.ReLU(),
            nn.Conv2d(32, 64, [2, 2]),
            nn.ReLU(),
            nn.Conv2d(64, 64, [2, 2]),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(1024, 64)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(64, num_outputs)
        self.value_fn = nn.Linear(64, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


class AdversarialPursuitModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(5, 32, [2, 2]),
            nn.ReLU(),
            nn.Conv2d(32, 64, [2, 2]),
            nn.ReLU(),
            nn.Conv2d(64, 64, [2, 2]),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(3136, 64)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(64, num_outputs)
        self.value_fn = nn.Linear(64, 1)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        if obs.shape[-1] != 5:
            obs = obs.reshape([obs.shape[0], 10, 10, 5])
        model_out = self.model(obs.permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


class AdversarialPursuitModelUnflat(AdversarialPursuitModel):
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        obs = obs.reshape([obs.shape[0], 10, 10, 5])
        model_out = self.model(obs.permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state


class BattleModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(5, 32, [2, 2]),
            nn.ReLU(),
            nn.Conv2d(32, 64, [2, 2]),
            nn.ReLU(),
            nn.Conv2d(64, 64, [2, 2]),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(6400, 64)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(64, num_outputs)
        self.value_fn = nn.Linear(64, 1)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        if obs.shape[-1] != 5:
            obs = obs.reshape([obs.shape[0], 13, 13, 5])
        model_out = self.model(obs.permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


class BattleModelUnflat(BattleModel):
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        obs = obs.reshape([obs.shape[0], 13, 13, 5])
        model_out = self.model(obs.permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state
