import argparse

import gym
import torch
import time
from gym.wrappers import AtariPreprocessing

import config
from utils import set_device, PongActionWrapper
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize
import multiprocessing

device = set_device()

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0', 'Pong-v0', 'CartPole-v1'])
parser.add_argument('--evaluate_freq', type=int, default=100, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=10, help='Number of evaluation episodes.', nargs='?')
parser.add_argument('--debug', type=bool, default=False, help='Enable debug mode to get more verbose output.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole,
    'Pong-v0': config.Pong,
    'CartPole-v1': config.CartPole
}


def run():
    args = parser.parse_args()
    print(f'Device type in use: {device}')

    if args.debug:
        print('Parallel info: ' + torch.__config__.parallel_info())

    cores = multiprocessing.cpu_count()
    torch.set_num_threads(cores)

    # Initialize environment and config.
    env = gym.make(args.env)

    env_config = ENV_CONFIGS[args.env]
    env_name = env.env.spec.id

    env = AtariPreprocessing(env, screen_size=env_config['screen_size'], grayscale_obs=True, frame_skip=1, noop_max=30)

    # Wrapper for eliminating extra actions from the agent.
    env = PongActionWrapper(env)

    # Initialize deep Q-network.
    dqn = DQN(env_config=env_config, env_name=env_name).to(device)

    # Initialize the target deep Q-network.
    dqn_target = DQN(env_config=env_config, env_name=env_name).to(device)

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of the best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    start_time = time.time()
    total_time = 0
    total_time_steps = 0

    for episode in range(env_config['n_episodes']):
        if episode > 0:
            now = time.time()
            elapsed = now - start_time
            start_time = now

            total_time += elapsed
            total_time_steps += 1
            print(f'episode: {episode}, elapsed time: {elapsed} s, mean time: {total_time / total_time_steps} s')
        else:
            print('episode:', episode)

        # The first observation.
        first_state = torch.tensor(env.reset(), device=device, dtype=torch.float32)

        # We begin by stacking the first observation 4 times.
        obs_stack = torch.cat(env_config["obs_stack_size"] * [first_state]).unsqueeze(0).to(device)
        done = False

        obs = first_state

        step = -1

        while not done:
            step += 1
            action = dqn.act(obs_stack, exploit=False)  # Get the action.
            prev_obs = obs
            obs, reward, done, info = env.step(action.item())  # Perform the next step.

            prev_obs_stack = obs_stack

            # Preprocess incoming observation.
            if not done:
                obs = torch.tensor(obs, device=device)
                # Add the new observation to the stack, and get one old observation.
                obs_stack = torch.cat((obs_stack[:, 84:, ...], obs.unsqueeze(0)), dim=1).to(device)
            # No observation if the episode is finished
            else:
                obs = None
                break  # This is what fixed our bug.

            # Push the current values to the memory.
            reward = torch.tensor(reward, device=device)
            memory.push(prev_obs_stack, action.squeeze(), obs_stack, reward)

            if step % env_config["train_frequency"] == 0:
                loss = optimize(dqn, target_dqn=dqn_target, memory=memory, optimizer=optimizer)

            if step % env_config["target_update_frequency"] == 0:
                dqn_target.copy_from(dqn)

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)

            print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.

            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best.pt')

    # Close environment after training is completed.
    env.close()


if __name__ == '__main__':
    run()
