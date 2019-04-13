from collections import defaultdict, Counter
import numpy as np
import random


"""
A rather stupid game just to test the concept of RL:
- one action leads to moving forward and get a reward
- one action leads to staying in place and get a penalty

!!! Important note !!!
It is not legal to have different rewards if the target state is the same (reward is attached to transition)
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
        return self.steps_left <= 0

    def step(self, action):
        if self.is_done():
            raise Exception("Game is over")
        if self.steps_left % 2 == action % 2:
            self.steps_left -= 1
            return 1
        else:
            return -1


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
            [0, -50, 1, 0],     # If you put 100 in first column, it still works :)
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
Same game, but the agent can slip to the bottom
"""


class SlipperyFindYourWayEnv(FindYourWayEnv):
    def __init__(self, slip_prob=0.2):
        super().__init__()
        self.slippy_prob = slip_prob

    def step(self, action):
        if action == 0 and 1 in self.get_actions():
            if random.random() < self.slippy_prob:
                action = 1
        return super().step(action)


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
        self.actions = defaultdict(set)             # map state to possible actions at that state
        self.transitions = defaultdict(Counter)     # map tuple (state, action) to expected target states (with prob)
        self.rewards = defaultdict(float)           # map tuple (state, action, new_state) to reward
        self.q_values = defaultdict(float)          # map tuple (state, action) to expected value
        self.temperature = 1.                       # controls the number of random actions attempted
        self.discount = 0.9                         # discount factor used in Q-learning bellman update

    def step(self, env) -> float:
        state = env.get_state()
        actions = env.get_actions()
        self.actions[state] |= set(actions)         # TODO - remove the | => state is supposed to be fully determined
        action = self._select_best(state, actions)
        reward = env.step(action)
        self._value_iteration(state, action, env.get_state(), reward)
        return reward

    def _select_best(self, state, actions):
        """
        Selects the action with the best Q-value (expected long term reward)
        """
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

    def _value_iteration(self, state, action, new_state, reward):
        """
        Update the Q-value table based on the transition and reward obtained by the action
        !! It is not as simple as updating based on target state (the transitions are random) !!
        """
        self.rewards[(state, action, new_state)] = reward
        self.transitions[(state, action)][new_state] += 1

        expected_reward = 0.
        expected_value = 0.
        total_transitions = sum(self.transitions[(state, action)].values())
        for new_state, count in self.transitions[(state, action)].items():
            max_next_value = max((self.q_values[(new_state, action)] for action in self.actions[new_state]), default=0)
            expected_reward += (count / total_transitions) * self.rewards[(state, action, new_state)]
            expected_value += (count / total_transitions) * max_next_value
        self.q_values[(state, action)] = expected_reward + self.discount * expected_value

    def temperature_decrease(self):
        self.temperature -= 0.1
        self.temperature = max(self.temperature, 0.)

    def __str__(self):
        return str({'q_values': self.q_values, 'temperature': self.temperature})


# TODO - implement double update (to speed training)


"""
An agent that uses Deep Q-learning
"""


# TODO - implement it using pytorch


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
    print("Random agent:", np.mean(rewards))


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
            print("Epoch", epoch + 1, ":", np.mean(rewards))
            agent.temperature_decrease()
            rewards.clear()


test_random_agent(env=LinearEnvironment())
print("-" * 20)
train_q_agent(env=LinearEnvironment())
print("-" * 20)
test_random_agent(env=FindYourWayEnv())
print("-" * 20)
train_q_agent(env=FindYourWayEnv())
print("-" * 20)
test_random_agent(env=SlipperyFindYourWayEnv(slip_prob=0.2))
print("-" * 20)
train_q_agent(env=SlipperyFindYourWayEnv(slip_prob=0.2))
