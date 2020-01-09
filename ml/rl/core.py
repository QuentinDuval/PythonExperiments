import abc
from collections import deque
import gym
import numpy as np
import sys
import time
import torch
from typing import *
import random


"""
Basic interface for any agent acting on an environment, in a given state
"""


class Agent(abc.ABC):
    @abc.abstractmethod
    def get_action(self, state):
        pass


class RandomAgent(Agent):
    def __init__(self, env: gym.Env):
        self.action_space = env.action_space

    def get_action(self, state):
        return self.action_space.sample()


"""
Utilities for tracking progress
"""


g_iteration_nb = 0
g_previous_time = None


def print_progress(done, total):
    global g_previous_time, g_iteration_nb
    g_iteration_nb += 1
    current_time = time.time_ns()
    if g_previous_time is not None:
        delay_ms = (current_time - g_previous_time) / 1_000_000
        if delay_ms > 100:
            throughput = g_iteration_nb / delay_ms * 1_000
            sys.stdout.write("\x1b[A") # Clear the line
            sys.stdout.write("{0}/{1} ({2:.2f}%) - {3:.2f} it/s".format(done, total, 100*done/total, throughput))
            sys.stdout.write("\r")
            g_iteration_nb = 0
            g_previous_time = time.time_ns()
    else:
        g_previous_time = time.time_ns()
    

def prange(end_range: int):
    print_progress(0, end_range)
    for i in range(end_range):
        print_progress(i+1, end_range)
        yield i
    sys.stdout.write("\n")


"""
A way to log what happens inside an environment
"""


class RecordingEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observations = []
        self.actions = []

    def reset(self, **kwargs):
        while len(self.observations) > len(self.actions):
            self.observations.pop()
        observation = self.env.reset(**kwargs)
        self.observations.append(observation)
        return observation

    def step(self, action):
        self.actions.append(action)
        observation, reward, done, info = self.env.step(action)
        if not done:
            self.observations.append(observation)
        return observation, reward, done, info


"""
A way to try an agent on a given environment
"""


def try_agent_on(env: gym.Env, agent: Agent, show=True) -> float:
    total_reward = 0.0
    obs = env.reset()
    if show:
        env.render()
    done = False
    while not done:
        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if show:
            env.render()
    if show:
        print("Total reward {0:.2f}".format(total_reward))
    return total_reward


def evaluate_performance(env: gym.Env, agent: Agent, episodes: int = 100):
    total_reward = 0.0
    for _ in range(episodes):
        total_reward += try_agent_on(env, agent, show=False)
    print("Average performance:", total_reward / episodes)


"""
Replay buffer
"""


class ReplayBuffer:
    def __init__(self, max_size: int):
        self.fifo = deque(maxlen=max_size)

    def add(self, observation: np.ndarray, action: int, value: float):
        self.fifo.append((observation, action, value))

    def sample(self, size) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        observations = []
        actions = []
        values = []
        for observation, action, value in random.choices(self.fifo, k=size):
            observations.append(torch.FloatTensor(observation))
            actions.append(action)
            values.append(value)
        return torch.stack(observations), torch.LongTensor(actions), torch.FloatTensor(values)

    def __len__(self):
        return len(self.fifo)

    
"""
Prioritized replay buffer
"""


class PrioritizedReplayBuffer:
    def __init__(self, max_size: int):
        self.fifo = deque(maxlen=max_size)
        self.weights = deque(maxlen=max_size)
    
    def add(self, observation: np.ndarray, action: int, prev_value: float, value: float):
        self.fifo.append((observation, action, value))
        self.weights.append(abs(value - prev_value))

    def sample(self, size) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        observations = []
        actions = []
        values = []
        for observation, action, value in random.choices(self.fifo, weights=self.weights, k=size):
            observations.append(torch.FloatTensor(observation))
            actions.append(action)
            values.append(value)
        return torch.stack(observations), torch.LongTensor(actions), torch.FloatTensor(values)
    
    def __len__(self):
        return len(self.fifo)
