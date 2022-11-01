import gym
import torch


def preprocess(obs, env):
    """Performs necessary observation preprocessing."""
    if env in ['CartPole-v0', 'Pong-v0', 'CartPole-v1']:
        return torch.tensor(obs, device=set_device()).float()
    else:
        raise ValueError('Please add necessary observation preprocessing instructions to preprocess() in utils.py.')


def set_device():
    """Sets the device for the project. Intended as an easy place to change the device that is in use."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


class PongActionWrapper(gym.ActionWrapper):
    """ A wrapper for limiting the action space of the Pong agent to only actions 2 and 3. """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = [2, 3]

    def action(self, action):
        return action
