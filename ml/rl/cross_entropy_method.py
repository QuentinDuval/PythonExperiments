"""
Cross Entropy Method:
- Start with a random policy
- Play N episodes with the current policy
- Take the episodes above a reward boundary (typically percentile 70th)
- Train on these "Elite" episodes (throw away the uninteresting ones)
=> Look like a kind of genetic algorithm stuff
"""

from collections import *
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from ml.rl.core import *

Episode = namedtuple('Episode', 'states actions score')


class Policy(abc.ABC):
    def get_action(self, state) -> 'action':
        pass

    def improve(self, episodes: List[Episode]):
        pass


class CrossEntropyAgent(Agent):
    def __init__(self, policy: Policy):
        self.policy = policy

    def get_action(self, env, state):
        return self.policy.get_action(state)

    def fit(self, env, max_iteration: int = 100, batch_size: int = 1000, batch_threshold: float = 0.7) -> List[float]:
        mean_scores = []
        for _ in range(max_iteration):
            episodes = [self._play_episode(env) for _ in range(batch_size)]
            mean_score = np.mean([episode.score for episode in episodes])
            mean_scores.append(mean_score)
            print("Mean score:", mean_score)
            if mean_score >= 200:
                print("Solved!")
                break

            episodes.sort(key=lambda episode: episode.score)
            episodes = episodes[int(batch_threshold * batch_size):]
            self.policy.improve(episodes)
        return mean_scores

    def _play_episode(self, env) -> List[Episode]:
        states, actions = [], []
        total_reward = 0.0
        state = env.reset()
        done = False
        while not done:
            action = self.get_action(env, state)
            states.append(state)
            actions.append(action)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        return Episode(states, actions, total_reward)


"""
Implementation of a policy to learn via a Neural Net
"""


class FullyConnectedNet(nn.Module):
    def __init__(self, observation_size, hidden_size, action_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, observations, with_softmax=False):
        ys = self.fc(observations)
        if with_softmax:
            return self.softmax(ys)
        return ys


class NeuralNetPolicy(Policy):
    def __init__(self, iteration_nb: int, learning_rate: float, net: nn.Module):
        self.net = net
        self.iteration_nb = iteration_nb
        self.learning_rate = learning_rate

    def get_action(self, state) -> 'action':
        self.net.eval()
        xs = torch.FloatTensor(state)
        ys = self.net(xs, with_softmax=True)
        probabilities = ys.detach().numpy()
        action = np.random.choice(len(probabilities), p=probabilities)
        return action

    def improve(self, episodes: List[Episode]):
        self.net.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        data_set = self._to_data_set(episodes)
        loader = data.DataLoader(data_set, batch_size=100, shuffle=True)
        for _ in range(self.iteration_nb):
            for states, actions in loader:
                optimizer.zero_grad()
                got = self.net(states, with_softmax=False)
                loss = criterion(got, actions)
                loss.backward()
                optimizer.step()

    def _to_data_set(self, episodes: List[Episode]):
        xs, ys = [], []
        for episode in episodes:
            for state, action in zip(episode.states, episode.actions):
                xs.append(state)
                ys.append(action)
        xs = torch.FloatTensor(xs)
        ys = torch.LongTensor(ys)
        return data.TensorDataset(xs, ys)
