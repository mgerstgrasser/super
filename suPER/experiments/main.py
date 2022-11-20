import argparse
import copy
import datetime
import os
from time import sleep
from uuid import uuid4

import numpy as np
import ray
from ray import tune
from ray.air.callbacks.wandb import WandbLoggerCallback

from suPER.experiments.config import *
from suPER.experiments.env import *


def parse_args(arg_string=None):
    parser = argparse.ArgumentParser()

    # We use a string type here, because argparse does not handle bools correctly.
    # This also allows us to specify different variations in the future.
    parser.add_argument(
        "--suPER",
        type=str,
        default="True",
        help="Set to 'True' to Enable the suPER mechanism",
    )
    parser.add_argument(
        "--suPER_insertion_method",
        type=str,
        default="multiagent",
        choices=["multiagent", "underlying"],
        help="Insert shared experiences as 'multiagent' batch or into 'underlying' buffer",
    )
    parser.add_argument(
        "--suPER_td_error_method",
        type=str,
        default="rllib",
        choices=["rllib", "ours"],
        help="Use 'rllib' builtin td error method or 'ours'.",
    )
    parser.add_argument(
        "--suPER_broadcast_heuristic",
        type=str,
        default="batch_single",
        choices=["stddev", "variance", "fixedbandwidth", "batch_single"],
        help="Use 'batch_single' broadcast heuristic or, 'variance', 'stddev', 'fixedbandwidth'.",
    )
    parser.add_argument(
        "--suPER_mode",
        type=str,
        default="gaussian",
        help="Use quantile, gaussian or stochastic mode",
    )
    parser.add_argument(
        "--suPER_bandwidth",
        type=float,
        default=0.01,
        help="Bandwidth for suPER as steps/step",
    )
    parser.add_argument(
        "--suPER_window",
        type=int,
        required=False,
        help="Window size for calculating mean/var/quantile in suPER",
    )

    parser.add_argument(
        "--team_sharing",
        type=str,
        default="",
        help="if team_sharing not empty, only agent whose policy id contains the string can share",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="seed for the environment",
    )

    parser.add_argument(
        "--num_seeds",
        type=int,
        default=1,
        help="number of seeds to train in parallel",
    )

    parser.add_argument(
        "--parameter_sharing",
        type=str,
        default="False",
        help="if parameter_sharing is True, only one policy is used for all agents",
    )

    parser.add_argument(
        "--fully_connected",
        type=str,
        default="False",
        help="if fully_connected is True, training is perform using fully connected network",
    )

    parser.add_argument("--batch_size", type=int, required=False, help="Override train batch size and rollout fragment length?")
    parser.add_argument("--train_batch_size", type=int, required=False, help="Override train batch size only?")
    parser.add_argument("--rollout_fragment_length", type=int, required=False, help="Override rollout fragment length only?")
    parser.add_argument("--num_timesteps", type=int, required=False, help="Override training length (timesteps)?")
    parser.add_argument("--lr", type=float, required=False, help="Override learning rate?")
    parser.add_argument("--buffer_size", type=int, required=False, help="Override replay buffer size?")

    # arguments for foraging environment
    parser.add_argument("--env_foraging_num_agents", type=int, required=False, help="Override number of foraging agents?")
    parser.add_argument("--env_foraging_board_size", type=int, required=False, help="Override foraging borasd size?")
    parser.add_argument("--env_foraging_food", type=int, required=False, help="Override number of foraging food in env?")
    parser.add_argument("--env_foraging_force_coop", type=int, required=False, help="Override coop settings of foraging?")
    parser.add_argument("--env_foraging_horizon", type=int, required=False, help="Override horizon settings of foraging?")
    parser.add_argument("--env_foraging_max_player_level", type=int, required=False, help="Override max players settings of foraging?")
    parser.add_argument("--env_foraging_po", type=int, required=False, help="Override partial observability settings of foraging?")

    # arguments for rware environment
    parser.add_argument("--env_rware_num_agents", type=int, required=False, help="Override number of rware agents?")
    parser.add_argument("--env_rware_rware_size", type=str, required=False, help="Override size of rware environment?")
    parser.add_argument("--env_rware_rware_modifier", type=int, required=False, help="Override rware modifier?")

    # arguments for harvest environment
    parser.add_argument("--env_harvest_num_agents", type=int, required=False, help="Override number of harvest agents?")
    parser.add_argument("--env_harvest_collective_reward", type=int, required=False, help="Override does use collective reward in harvest env?")
    parser.add_argument("--env_harvest_num_switches", type=int, required=False, help="Override num switches in harvest env?")

    # arguments for cleanup environment
    parser.add_argument("--env_cleanup_num_agents", type=int, required=False, help="Override number of cleanup agents?")
    parser.add_argument("--env_cleanup_collective_reward", type=int, required=False, help="Override does use collective reward in cleanup env?")
    parser.add_argument("--env_cleanup_num_switches", type=int, required=False, help="Override num switches in cleanup env?")

    # arguments for pursuit environment
    parser.add_argument("--env_pursuit_num_agents", type=int, required=False, help="Override number of pursuit agents?")
    parser.add_argument("--env_pursuit_n_evaders", type=int, required=False, help="Override num of evaders in pursuit env?")
    parser.add_argument("--env_pursuit_shared_reward", type=int, required=False, help="Override shared reward setting in pursuit env?")

    # arguments for Atari environment
    parser.add_argument("--env_atari_po", type=int, required=False, help="Make Atari partially observable?")
    parser.add_argument("--env_atari_randomstart", type=int, required=False, help="Make Atari random start?")

    # arguments for battle environment
    parser.add_argument("--env_battle_map_size", type=int, required=False, help="Override map size for battle?")

    # arguments for adversarial pursuit environment
    parser.add_argument("--env_adversarial_pursuit_map_size", type=int, required=False, help="Override map size for adversarial pursuit?")

    parser.add_argument(
        "--battle_advpursuit_against_pretrained",
        action="store_true",
        help="If enabled, init ray in local mode.",
    )

    parser.add_argument("--exploration_initial_epsilon", type=float, required=False, help="Override initial epsilon in DQN?")
    parser.add_argument("--exploration_final_epsilon", type=float, required=False, help="Override final epsilon in DQN?")

    parser.add_argument("--parameter_tunning", type=str, required=False, help="if true, grid_search is used for hyperparameter tunning during training")

    parser.add_argument(
        "--env",
        type=str,
        default="pursuit",
        help="Determines the training environment",
    )

    parser.add_argument(
        "--num_cpus",
        type=int,
        default=1,
        help="Number of CPU cores",
    )
    parser.add_argument(
        "--experiment_group",
        type=str,
        default="suPER-test",
        help="Run group to use for logging",
    )
    parser.add_argument(
        "--experiment_project",
        type=str,
        default="suPER",
        help="wandb project to use for logging",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=f"experiment_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}",
        help="Run name to use for logging",
    )
    parser.add_argument(
        "--ray_local_mode",
        action="store_true",
        help="If enabled, init ray in local mode.",
    )
    parser.add_argument(
        "--ray_plain_init",
        action="store_true",
        help="If enabled, init ray without arguments.",
    )
    parser.add_argument(
        "--eval_in_parallel",
        action="store_true",
        help="If enabled, evaluate in parallel to training, using extra worker.",
    )
    parser.add_argument(
        "--ray_address",
        type=str,
        required=False,
        help="Address of Ray cluster to connect to, if not set start local cluster.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=False,
        help="Filename with experiments to run, one per line, as CLI arg strings. Most other args are ignored if this is set.",
    )
    parser.add_argument(
        "--use_gpu",
        type=float,
        required=False,
        help="How many GPUs to use, if any",
    )

    parser.add_argument(
        "--evaluation",
        type=str,
        default="False",
        help="if True, enable evaluation settings",
    )

    parser.add_argument("--framework", type=str, required=False, help="Override framework to use?")

    parser.add_argument(
        "--horizon",
        type=int,
        default=500,
        help="Maximum time steps for each episode in the environment",
    )

    args, remaining_cli = parser.parse_known_args(arg_string)

    return args, remaining_cli


def get_trainer_and_config(args):
    """Takes the cli arguments, and returns the trainer and config to use."""
    if args.env == "battle":
        trainer, config, stop = config_battlev4_dqn(args)

    elif args.env == "battle_plaindqn":
        trainer, config, stop = config_battlev4_plaindqn(args)

    elif args.env == "battle_maddpg":
        trainer, config, stop = config_battlev4_maddpg(args)

    elif args.env == "battle_qmix":
        trainer, config, stop = config_battlev4_qmix(args)

    elif args.env == "adversarial_pursuit":
        trainer, config, stop = config_adversarial_pursuit_dqn(args)

    elif args.env == "adversarial_pursuit_plaindqn":
        trainer, config, stop = config_adversarial_pursuit_plaindqn(args)

    elif args.env == "adversarial_pursuit_maddpg":
        trainer, config, stop = config_adversarial_pursuit_maddpg(args)

    elif args.env == "adversarial_pursuit_qmix":
        trainer, config, stop = config_adversarial_pursuit_qmix(args)

    elif args.env == "pursuit":
        trainer, config, stop = config_pursuit_dqn(args)

    elif args.env == "pursuit_plaindqn":
        trainer, config, stop = config_pursuit_plaindqn(args)

    elif args.env == "pursuit_maddpg":
        trainer, config, stop = config_pursuit_maddpg(args)

    elif args.env == "pursuit_qmix":
        trainer, config, stop = config_pursuit_qmix(args)
    else:
        raise NotImplementedError(f"Environment {args.env} not implemented.")

    # Enable or disable suPER, but only for configs that support it, i.e. already have it.
    if "suPER" in config:
        config["suPER"] = True if args.suPER == "True" else False
        config["suPER_insertion"] = args.suPER_insertion_method
        config["suPER_td_error"] = args.suPER_td_error_method
        config["suPER_mode"] = args.suPER_mode
        config["suPER_bandwidth"] = args.suPER_bandwidth
        if args.team_sharing is not None:
            config["suPER_team_sharing"] = args.team_sharing
        if args.suPER_window is not None:
            config["suPER_mean_stddev_window"] = args.suPER_window

    # Hack to save CLI args to wandb
    if "input_config" in config:
        config["input_config"]["cli_arg_dict"] = vars(args)
    else:
        config["input_config"] = {"cli_arg_dict": vars(args)}

    # Allow overriding train batch size and rollout fragment length, either together or separately.
    if args.batch_size is not None:
        config["train_batch_size"] = args.batch_size
        config["rollout_fragment_length"] = args.batch_size
    if args.train_batch_size is not None:
        config["train_batch_size"] = args.train_batch_size
    if args.rollout_fragment_length is not None:
        config["rollout_fragment_length"] = args.rollout_fragment_length
    if args.lr is not None:
        config["lr"] = args.lr
    if args.buffer_size is not None:
        config["buffer_size"] = args.buffer_size
    if args.horizon is not None:
        config["horizon"] = args.horizon

    config["disable_env_checking"] = True

    if args.use_gpu is not None:
        config["num_gpus"] = args.use_gpu

    if args.num_timesteps is not None:
        stop = {"timesteps_total": args.num_timesteps}

    if args.framework is not None:
        config["framework"] = args.framework
    if config["framework"] in ["tf2", "tfe"]:
        config["eager_tracing"] = True

    if args.exploration_initial_epsilon is not None:
        config["exploration_config"]["initial_epsilon"] = args.exploration_initial_epsilon
    if args.exploration_final_epsilon is not None:
        config["exploration_config"]["final_epsilon"] = args.exploration_final_epsilon

    config["seed"] = args.seed
    config["env_config"]["seed"] = args.seed + 100

    if args.eval_in_parallel:
        config["evaluation_parallel_to_training"] = True
        config["evaluation_num_workers"] = 1
        config["evaluation_interval"] = 1
        config["evaluation_duration"] = "auto"
        if "battle" in config["env"]:
            config["evaluation_num_workers"] = 3
            config["min_sample_timesteps_per_iteration"] *= 4

    return trainer, config, stop


def try_start_ray(num_cpus, local_mode, depth=0):
    try:
        waittime = np.random.randint(1, 10 * 2**depth)
        print(f"Will try to start ray after sleeping for {waittime} seconds.")
        sleep(waittime)
        print("Trying to start ray.")
        ray.init(num_cpus=num_cpus, local_mode=local_mode, include_dashboard=False)
    except:
        print(f"Failed to start ray on attempt {depth+1}. Retrying...")
        try_start_ray(num_cpus, local_mode, depth + 1)


def main(args, num_cpus, group: str = "suPER", name: str = f"experiment_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}", ray_local_mode: bool = False):
    if args.ray_plain_init:
        ray.init()
    elif args.ray_address is None:
        try_start_ray(num_cpus, ray_local_mode, 0)
    else:
        ray.init(address=args.ray_address, include_dashboard=False)

    tune.register_env("pursuit", lambda config: env_creator_pursuit(config))
    tune.register_env("battle", lambda config: env_creator_battle(config))
    tune.register_env("adversarial_pursuit", lambda config: env_creator_adversarial_pursuit(config))

    curr_folder = os.path.dirname(os.path.realpath(__file__))
    local_dir = curr_folder + "/ray_results/" + uuid4().hex + "/"

    # Set up Weights And Biases logging if API key is set in environment variable.
    if "WANDB_API_KEY" in os.environ:
        callbacks = [
            WandbLoggerCallback(
                project=args.experiment_project,
                api_key=os.environ["WANDB_API_KEY"],
                log_config=True,
                resume=False,
                group=group,
                # name=name,
                # id=f"{args.experiment_name}_{os.environ.get('SEED', 42)}_{datetime.datetime.now().timestamp()}_{os.getpid()}",
            )
        ]
    else:
        callbacks = []
        print("No wandb API key specified, running without wandb logging.")

    experiments = []

    if args.config_file is None:
        trainer, config, stop = get_trainer_and_config(args)
        for i in range(args.num_seeds):
            this_config = copy.deepcopy(config)
            this_config["seed"] = config.get("seed", 0) + i
            this_config["env_config"]["seed"] = config.get("seed", 0) + i + 100
            this_config["input_config"]["name"] = f"exp!{this_config['env']}!{args.experiment_group}!{this_config['seed']}"
            this_config["input_config"]["group"] = args.experiment_group
            exp = tune.Experiment(
                f"exp!{this_config['env']}!{args.experiment_group}!{this_config['seed']}",
                run=trainer,
                config=this_config,
                stop=stop,
                local_dir=local_dir,
                # checkpoint_at_end=True,
                # checkpoint_freq=25,
                max_failures=3,
            )
            experiments.append(exp)
    else:
        # Read file into list of strings:
        with open(args.config_file, "r") as f:
            config_list = f.readlines()
        for config_str in config_list:
            if config_str[0] == "#":
                continue
            these_args, _ = parse_args(config_str.split())
            trainer, config, stop = get_trainer_and_config(these_args)
            for i in range(args.num_seeds):
                this_config = copy.deepcopy(config)
                this_config["seed"] = config.get("seed", 0) + i
                this_config["env_config"]["seed"] = config.get("seed", 0) + i + 100
                this_config["input_config"]["name"] = f"exp!{this_config['env']}!{these_args.experiment_group}!{this_config['seed']}"
                this_config["input_config"]["group"] = these_args.experiment_group
                exp = tune.Experiment(
                    f"exp!{this_config['env']}!{these_args.experiment_group}!{this_config['seed']}",
                    run=trainer,
                    config=this_config,
                    stop=stop,
                    local_dir=local_dir,
                    # checkpoint_at_end=True,
                    # checkpoint_freq=25,
                    max_failures=3,
                )
                experiments.append(exp)

    tune.run(experiments, callbacks=callbacks, raise_on_failed_trial=False)

    print("Done with experiment. Shutting down ray.")

    # tuner = tune.Tuner(
    #     trainer,
    #     param_space=config,
    #     run_config=air.RunConfig(
    #         stop=stop,
    #         callbacks=callbacks,
    #         local_dir=local_dir,
    #         checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True, checkpoint_frequency=25),
    #     ),
    # )
    # tuner.fit()

    # ray.shutdown()


if __name__ == "__main__":

    args, remaining_cli = parse_args()

    for a in remaining_cli:
        print(f"WARNING! Ignoring unknown argument {a}.")

    main(args, args.num_cpus, group=args.experiment_group, name=args.experiment_name, ray_local_mode=args.ray_local_mode)
