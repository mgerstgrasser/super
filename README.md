# suPER - Shared Multi-Agent Prioritized Experience Replay

Matthias Gerstgrasser, Tom Danino, Sarah Keren

[suPER](https://matthias.gerstgrasser.net/files/suPER-preprint.pdf) is a multi-agent DQN algorithm, sharing high td-error experiences among agents similarly to how experiences are sampled from the replay buffer in (single-agent) DQN with PER.

## Installation

To install, run:

  ```bash
  pip install -e .
  ```

## Usage

suPER is built as a modified [RLlib DQN algorithm](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#dqn). In order to use suPER in your project, simply replace the `DQN` algorithm in your code with `suPERDQN` from our package. You can modify suPER settings using the `suPERDQNConfig.train()` method. For example:
  
  ```python
  from suPER.core.trainers import suPERDQN, suPERDQNConfig

  config = suPERDQNConfig().train(suPER_bandwidth=0.2).environment(...)
  trainer = suPERDQN(config=config)
  ```

## Citation

Coming soon! Please check back here in January.

## Reproducing Paper Experiments

We also provide implementations of all the experiments in the paper. To reproduce the experiments, follow these instructions:

### Installing Preqrequisites for Experiments

We recommend setting up a virtual Python environment using pyenv-virtualenv, Anaconda or similar, using Python version 3.8 or higher.
Installing the suPER package will install all dependencies, but due to a conflict between `ray` and `PettingZoo`, it is currently necessary
to install `gym` version 0.22 manually afterwards. The following should work. On Linux, you may have to `sudo apt install swig -y` first.

```bash
pip install -U tensorflow==2.10.0 tensorflow-probability==0.17.0 torch==1.12.1 wandb 'pettingzoo[magent,sisl]==1.20.1' supersuit==3.5.0
pip install -U gym==0.22.0
```

### Running suPER Experiments

To run experiments, run `suPER/experiments/main.py` with the appropriate arguments. Set the environment to one of `pursuit`, `battle`, or `adversarial_pursuit`. You can set `--seed` and `--num_seeds` to run one or more specific seeds.

```{bash}
main.py [--suPER SUPER] [--suPER_mode SUPER_MODE] [--suPER_bandwidth SUPER_BANDWIDTH] [--env ENV] 

options:
  -h, --help            show this help message and exit
  --suPER SUPER         Set to 'True' to Enable the suPER mechanism
  --suPER_mode SUPER_MODE
                        Use quantile, gaussian or stochastic mode
  --suPER_bandwidth SUPER_BANDWIDTH
                        Bandwidth for suPER as steps/step
  --suPER_window SUPER_WINDOW
                        Window size for calculating mean/var/quantile in suPER
  --team_sharing TEAM_SHARING
                        if team_sharing not empty, only agent whose policy id contains the string can share
  --env ENV             Determines the training environment
  --ray_local_mode      If enabled, init ray in local mode.
  --use_gpu USE_GPU     How many GPUs to use, if any
```

### Running SEAC Experiments

We provide a local copy of the SEAC code modified to work with the PettingZoo environments we use in the paper. This is under `seac/`. Run `python train.py with env=pursuit` and similar for `battle`, `adversarial_pursuit`.

### Ray Cluster

If you would like to run a large number of experiments, we provide a Ray cluster configuration for AWS, in ray_runtime.yaml. Follow the Ray setup instructions. Running `run_all.sh` will submit most experiments (except MADDPG and SEAC) to the Ray cluster.
