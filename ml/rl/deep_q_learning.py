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

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ml.rl.core import *

Experience = namedtuple('Experience', 'state action next_state reward done')
ExperienceBatch = namedtuple('ExperienceBatch', 'states actions next_states rewards dones')


class Experiences:
    def __init__(self, size: int):
        self.buffer = deque(maxlen=size)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        self.buffer.append(experience)

    def extend(self, experiences: List[Experience]):
        for experience in experiences:
            self.append(experience)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, next_states, rewards, dones = zip(*[self.buffer[idx] for idx in indices])
        return ExperienceBatch(
            states=np.array(states),
            actions=np.array(actions),
            next_states=np.array(next_states),
            rewards=np.array(rewards, dtype=np.float32),
            dones=np.array(dones, dtype=np.uint8))


class NeuralNetQValues:
    def __init__(self, net: nn.Module, iteration_nb: int, learning_rate: float, reward_discount: float = 1.):
        self.net = net
        self.target_net = copy.deepcopy(self.net)
        self.iteration_nb = iteration_nb
        self.reward_discount = reward_discount
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

    def best_action(self, state) -> 'action':
        self.net.eval()
        xs = torch.FloatTensor(state)
        ys = self.target_net(xs)
        return torch.argmax(ys, dim=-1).detach().numpy()

    def improve_valuation(self, experiences: ExperienceBatch):
        self.net.eval()
        states = torch.FloatTensor(experiences.states)
        actions = torch.LongTensor(experiences.actions)
        next_states = torch.FloatTensor(experiences.next_states)
        rewards = torch.FloatTensor(experiences.rewards)
        dones = torch.ByteTensor(experiences.dones)
        next_values = self.target_net(next_states).max(dim=-1)[0]
        next_values[dones] = 0.0  # Set all values of terminal states to zero
        expected_values = next_values * self.reward_discount + rewards

        self.net.train()
        self.optimizer.zero_grad()
        current_values = self._gather_action_values(self.net(states), actions)
        loss = self.criterion(current_values, expected_values)
        loss.backward()
        self.optimizer.step()

    def improve_policy(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def _gather_action_values(self, out_values, actions):
        action_indices = actions.unsqueeze(dim=-1)
        values = torch.gather(out_values, dim=-1, index=action_indices)
        return values.squeeze(dim=-1)


class DeepQLearningAgent(Agent):
    def __init__(self, q_values: NeuralNetQValues, start_epsilon: float = 0.1, min_epsilon: float = 0.0):
        self.q_values = q_values
        self.epsilon = start_epsilon
        self.min_epsilon = min_epsilon

    def get_action(self, env, state):
        if self.epsilon > 0. and np.random.random() < self.epsilon:
            return env.action_space.sample()
        return self.q_values.best_action(state)

    def fit(self, env,
            is_success,
            max_episodes: int = 100_000,
            replay_buffer_size: int = 2_000,
            replay_start_size: int = 1_000,
            batch_size: int = 100,
            policy_improvement_size: int = 1_000,
            epsilon_decrease_factor: float = 0.) -> List[float]:

        experiences = Experiences(replay_buffer_size)
        mean_scores = []
        scores = deque(maxlen=policy_improvement_size)

        for episode_id in range(1, max_episodes):
            episode_experiences, episode_score = self._play_episode(env)
            experiences.extend(episode_experiences)
            scores.append(episode_score)

            if is_success(episode_score):
                self.epsilon = max(self.min_epsilon, self.epsilon * epsilon_decrease_factor)

            if len(experiences) >= replay_start_size:
                self.q_values.improve_valuation(experiences.sample(batch_size))

            if episode_id % policy_improvement_size == 0:
                mean_scores.append(sum(scores) / len(scores))
                print("#", episode_id, "Mean score:", mean_scores[-1], "(epsilon:", self.epsilon, ")")
                self.q_values.improve_policy()

        return mean_scores

    def _play_episode(self, env):
        experiences = []
        total_reward = 0.0
        state = env.reset()
        done = False
        while not done:
            action = self.get_action(env, state)
            next_state, reward, done, _ = env.step(action)
            experiences.append(Experience(state, action, next_state, reward, done))
            total_reward += reward
            state = next_state
        return experiences, total_reward
