import argparse
import importlib
import os
import sys

import numpy as np
import torch as th
import yaml
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.utils import set_random_seed

import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.load_from_hub import download_from_hub
from rl_zoo3.utils import StoreDict, get_model_path

SAVE_PATH = "./bc_data"

def generate_expert_traj(model, env, args, save_path, n_episodes=1, image_folder=None, sticky_action_prob=0.25, random_initial_steps=0):
    stochastic = not args.deterministic
    deterministic = not stochastic
    episode_start = np.ones((env.num_envs,), dtype=bool)

    obs_space = env.observation_space

    taken_actions = []
    observations = []
    rewards = []
    episode_returns = np.zeros((n_episodes,))
    episode_starts = []
    model_selected_actions = []
    repeated = []

    ep_idx = 0
    obs = env.reset()
    episode_starts.append([True])
    reward_sum = 0.0
    idx = 0
    lstm_states = None

    prev_action = None
    random_steps = random_initial_steps > 0

    while ep_idx < n_episodes:
        if random_steps:
            for _ in range(random_initial_steps):
                obs, _, _, _ = env.step([env.action_space.sample()])
            # Set random steps = False until the next episode
            random_steps = False

        observations.append(obs)

        # Choose an action
        model_selected_action, lstm_states = model.predict(
                    obs,  # type: ignore[arg-type]
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=deterministic,
                )
        

        # With probability sticky_action_prob, the agent repeats the last
        # action, ignoring the chosen action above. Note that this
        # is implemented in the gym environment, but I do not know how that
        # interacts with all the stable-baselines code on top of it, so I am
        # doing it here. This also has the advantage in allowing us to
        # distinguish between repeated actions and normal actions.
        if prev_action is None or np.random.uniform() > sticky_action_prob:
            taken_action = model_selected_action
            repeated.append(False)
        else:
            taken_action = prev_action
            repeated.append(True)

        obs, reward, done, info = env.step(taken_action)
        prev_action = taken_action

        # Use only first env
        model_selected_action = np.array([model_selected_action[0]])
        taken_action = np.array([taken_action[0]])
        reward = np.array([reward[0]])
        done = np.array([done[0]])

        taken_actions.append(taken_action)
        model_selected_actions.append(model_selected_action)
        rewards.append(reward)
        episode_starts.append(done)
        reward_sum += reward
        idx += 1
        if done:
            obs = env.reset()
            # Reset the state in case of a recurrent policy
            state = None

            episode_returns[ep_idx] = reward_sum
            reward_sum = 0.0
            ep_idx += 1

            # Once done, set random_steps = True so that next sample starts off
            # with random actions.
            if random_initial_steps > 0:
                random_steps = True
    
    observations = np.array(observations)
    taken_actions = np.array(taken_actions)
    model_selected_actions = np.array(model_selected_actions)
    rewards = np.array(rewards)
    episode_starts = np.array(episode_starts)
    repeated = np.array(repeated)

    assert len(observations) == len(taken_actions)
    assert len(taken_actions) == len(model_selected_actions)

    numpy_dict = {
        'model selected actions': model_selected_actions,
        'taken actions': taken_actions,
        'obs': observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts,
        'repeated': repeated
    }

    print(f"Number of steps: {idx}")
    print(f"Number of observations: {len(observations)}")
    print(f"Number of taken actions: {len(taken_actions)}")

    if save_path is not None:
        np.savez(save_path, **numpy_dict)
    
    env.close()



def generate_trajectories(args, save_path, n_episodes=1, atari=False, suffix='', seed=None, sticky_action_prob=0.25, random_initial_steps=0):
    env_name: EnvironmentName = args.env
    algo = args.algo
    _, model_path, log_path = get_model_path(
            args.exp_id,
            args.folder,
            algo,
            env_name,
            args.load_best,
            args.load_checkpoint,
            args.load_last_checkpoint,
        )
    
    #print(f"Loading {model_path}")

    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    seed1 = np.random.randint(2**32 - 1, dtype="int64").item()

    set_random_seed(seed1)

    is_atari = ExperimentManager.is_atari(env_name.gym_id)
    is_minigrid = ExperimentManager.is_minigrid(env_name.gym_id)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    seed2 = np.random.randint(2**32 - 1, dtype="int64").item()

    env = create_test_env(
        env_name.gym_id,
        n_envs=args.n_envs,
        stats_path=maybe_stats_path,
        seed=seed2,
        log_dir=log_dir,
        should_render=False, #not args.no_render, ##
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    seed3 = np.random.randint(2**32 - 1, dtype="int64").item()

    kwargs = dict(seed=seed3)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))
        # Hack due to breaking change in v1.6
        # handle_timeout_termination cannot be at the same time
        # with optimize_memory_usage
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version or args.custom_objects:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    if "HerReplayBuffer" in hyperparams.get("replay_buffer_class", ""):
        kwargs["env"] = env

    model = ALGOS[algo].load(model_path, custom_objects=custom_objects, device=args.device, **kwargs)

    path = os.path.join(save_path, env_name)
    if not os.path.isdir(path):
        os.makedirs(path)
    image_folder = '{}-recorded_images-{}'.format(env_name, suffix)
    save_path = os.path.join(path, '{}_{}'.format(env_name, suffix))

    generate_expert_traj(
        model, env, args, save_path=save_path, n_episodes=n_episodes,
        image_folder=image_folder, sticky_action_prob=sticky_action_prob, random_initial_steps=random_initial_steps
    )


def generate_trajectories_multiple(n_episodes_per_folder, num_folders, args, save_path, atari=False, seed_list=None, sticky_action_prob=0.25, random_initial_steps=0):
    print("Starting generation...")
    if seed_list is not None:
        assert len(seed_list) == num_folders
    for i in range(0, num_folders):
        print('Starting batch of trajectories: {}'.format(i))
        if seed_list is None:
            seed = None
        else:
            seed = seed_list[i]
        if n_episodes_per_folder == 1:
            suffix = '{}'.format(seed)
        else:
            suffix = '{}_to_{}'.format(i*n_episodes_per_folder,
                                       (i+1)*n_episodes_per_folder-1)
        
        generate_trajectories(
            args, save_path, n_episodes=n_episodes_per_folder,
            atari=atari, suffix=suffix, seed=seed,
            sticky_action_prob=sticky_action_prob,
            random_initial_steps=random_initial_steps
        )

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=EnvironmentName, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument(
        "--custom-objects", action="store_true", default=False, help="Use custom objects to solve loading issues"
    )
    parser.add_argument(
        "-P",
        "--progress",
        action="store_true",
        default=False,
        help="if toggled, display a progress bar using tqdm and rich",
    )
    args = parser.parse_args()

    generate_trajectories_multiple(
        1, 200, args, save_path=os.path.join(SAVE_PATH, 'rand_steps=0'),
        atari=True, seed_list=range(200), sticky_action_prob=0.25
    )
    generate_trajectories_multiple(
        1, 50, args, save_path=os.path.join(SAVE_PATH, 'rand_steps=75'),
        atari=True, seed_list=range(10000, 10050), sticky_action_prob=0.25
    )
    generate_trajectories_multiple(
        1, 50, args, save_path=os.path.join(SAVE_PATH, 'rand_steps=100'),
        atari=True, seed_list=range(20000, 20050), sticky_action_prob=0.25
    )

if __name__ == "__main__":
    generate()