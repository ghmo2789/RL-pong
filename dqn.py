import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import set_device


device = set_device()


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config, env_name):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]

        self.env_name = env_name

        # Pong NN
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

        self.flattener = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        # Normalizing the values.
        normalizer = lambda y: y / 255
        x_normalized = normalizer(x)
        h1 = self.conv1(x_normalized)
        h1 = F.relu(h1)
        h2 = self.conv2(h1)
        h2 = F.relu(h2)
        h3 = self.conv3(h2)
        h3 = F.relu(h3)
        h4 = self.flattener(h3)
        h5 = self.fc1(h4)
        h5 = F.relu(h5)
        h6 = self.fc2(h5)

        return h6

    def copy_from(self, other: nn.Module):
        """ Replicates the neural network. """
        self.load_state_dict(other.state_dict())

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        self.eps_start *= 0.9999
        epsilon = max(self.eps_end, self.eps_start)

        if exploit or random.random() > epsilon:
            with torch.no_grad():
                observation = torch.reshape(observation, (1, 4, 84, 84))
                res = self(observation)
                best_action = torch.argmax(res)

                # Force the agent to return 2 or 3 as input choices.
                if best_action.item() == 0:
                    return torch.tensor(2, device=device, dtype=torch.long)
                else:
                    return torch.tensor(3, device=device, dtype=torch.long)
        else:
            # Pong only has 2 actions. The action for DOWN is number 2 and the action for UP is number 3.
            # The rest are useless to pong.
            if self.env_name == 'Pong-v0':
                pong_actions = [2, 3]
                return torch.tensor([[random.choice(pong_actions)]], device=device, dtype=torch.long)
            else:
                return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)


def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    states, action, next_states, reward = memory.sample(dqn.batch_size)

    state_tensor = torch.cat(states).to(device)  # Tensor of the states.

    state_tensor = torch.reshape(state_tensor, (32, 4, 84, 84))
    action = [a.unsqueeze(0) for a in action]
    # -2 is done to force the inputs (which are all 2 or 3) to conform to 0 or 1.
    action_tensor = torch.cat(action).view(dqn.batch_size, 1).to(device) - 2
    reward_tensor = torch.tensor(reward, device=device)

    result = dqn(state_tensor)
    q_values = result.gather(1, action_tensor)  # Q-values gotten from the results.

    # Filter out invalid states.
    invalid_states_filter = torch.tensor([state is not None for state in next_states], device=device)
    valid_next_states = [state for state in next_states if state is not None]

    valid_next_states_tensor = torch.cat(valid_next_states)
    valid_next_states_tensor = torch.reshape(valid_next_states_tensor, (32, 4, 84, 84)).to(device)
    action_values_tensors = torch.zeros(dqn.batch_size, device=device).to(device)

    # A tensor of the best action values.
    action_values_tensors[invalid_states_filter] = target_dqn(valid_next_states_tensor).max(1).values

    # update the q-values.
    q_value_targets = (action_values_tensors * dqn.gamma) + reward_tensor

    loss = F.mse_loss(q_values.squeeze(), q_value_targets).to(device)

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item()
