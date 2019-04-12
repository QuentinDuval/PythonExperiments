from collections import defaultdict
import numpy as np
import random


"""
A rather stupid game just to test the concept of RL:
- There are two actions, one strictly superior in any case, but with a random reward
- The environment evolves always the same, not depending on the action
"""


class LinearEnvironment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.steps_left = 10

    def get_state(self):
        return (self.steps_left,)

    def get_actions(self):
        return [1, 2]

    def is_done(self):
        return self.steps_left == 0

    def step(self, action):
        if self.is_done():
            raise Exception("Game is over")
        reward = self.reward(action)
        self.steps_left -= 1
        return reward

    def reward(self, action):
        if self.steps_left % 2 == 0:
            # return random.randint(1, action)
            return action
        else:
            # return random.randint(-action, -1)
            return -action


"""
A game in which you are supposed to find the top corner of the map
- There are positions in which you gain some power pills
- There are positions in which you just loose
"""


class FindYourWayEnv:
    def __init__(self):
        self.map = [
            [0, 0, 0, 0],
            [-50, 1, 1, 0],
            [0, -50, 1, 0],
            [0, 0, 1, 100]
        ]
        self.h = len(self.map)
        self.w = len(self.map[0])
        self.reset()

    def reset(self):
        self.i = 0
        self.j = 0

    def get_state(self):
        return (self.i, self.j)

    def get_actions(self):
        actions = []
        if self.i < self.h - 1:
            actions.append(0)
        if self.j < self.w - 1:
            actions.append(1)
        return actions

    def is_done(self):
        return self.i == self.h - 1 and self.j == self.w - 1

    def step(self, action):
        if self.is_done():
            raise Exception("Game is over")
        if action == 0:
            self.i += 1
        elif action == 1:
            self.j += 1
        reward = self.map[self.i][self.j]
        return reward


"""
An agent that acts randomly
"""


class RandomAgent:
    def step(self, env) -> float:
        state = env.get_state()
        actions = env.get_actions()
        reward = env.step(random.choice(actions))
        return reward


"""
An agent that uses Q-learning
"""


class QAgent:
    def __init__(self):
        self.q_values = defaultdict(float)  # q_values maps tuple (state, action) to expected value
        self.temperature = 1.               # controls the number of random actions attempted
        self.discount = 0.9                 # discount factor used in Q-learning bellman update

    def step(self, env) -> float:
        state = env.get_state()
        actions = env.get_actions()
        action = self.select_best(state, actions)
        reward = env.step(action)
        self.value_iteration(state, action, reward, env)
        return reward

    def select_best(self, state, actions):
        if random.random() < self.temperature:
            return random.choice(actions)

        best_action = None
        best_action_reward = -1 * float('inf')
        for action in actions:
            reward = self.q_values[(state, action)]
            if reward > best_action_reward:
                best_action = action
                best_action_reward = reward
        return best_action

    def value_iteration(self, state, action, reward, env):
        new_state = env.get_state()
        max_next_state = max((self.q_values[(new_state, action)] for action in env.get_actions()), default=0)
        self.q_values[(state, action)] = reward + self.discount * max_next_state

    def temperature_decrease(self):
        self.temperature -= 0.1
        self.temperature = max(self.temperature, 0.)

    def __str__(self):
        return str({'q_values': self.q_values, 'temperature': self.temperature})


"""
Testing the different models
"""


def test_random_agent(env):
    agent = RandomAgent()

    rewards = []
    for epoch in range(100):
        total_reward = 0.
        env.reset()
        while not env.is_done():
            total_reward += agent.step(env)
        rewards.append(total_reward)
    print(np.mean(rewards))


def train_q_agent(env):
    agent = QAgent()

    rewards = []
    for epoch in range(2000):
        total_reward = 0.
        env.reset()
        while not env.is_done():
            total_reward += agent.step(env)
        rewards.append(total_reward)
        if (1 + epoch) % 100 == 0:
            print(np.mean(rewards))
            agent.temperature_decrease()
            rewards.clear()


test_random_agent(env=LinearEnvironment())
print("-" * 20)
train_q_agent(env=LinearEnvironment())
print("-" * 20)
test_random_agent(env=FindYourWayEnv())
print("-" * 20)
train_q_agent(env=FindYourWayEnv())
