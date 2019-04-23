from collections import defaultdict, Counter
import copy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim


"""
A rather stupid game just to test the concept of RL:
- one action leads to moving forward and get a reward
- one action leads to staying in place and get a penalty

!!! Important note !!!
- It's not legal to have different rewards if state, action, target state is the same (reward is attached to transition)
- BUT, if the state contains something that allows the distinction (example: random seed), it is okay
"""


class LinearEnvironment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.steps_left = 10

    def sample(self):
        self.steps_left = random.randint(1, 10)

    def get_state(self):
        return self.steps_left

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
A game in which you are supposed to find the bottom-right corner of the map
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

    def sample(self):
        self.i = random.randint(0, self.h - 1)
        self.j = random.randint(0, self.w - 1)

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
        self.slip_prob = slip_prob

    def step(self, action):
        if action == 0 and 1 in self.get_actions():
            if random.random() < self.slip_prob:
                action = 1
        return super().step(action)


"""
A game in which you are supposed to collect as much coins as possible
- There are positions in which you gain coins
- There are positions in which you loose coins (robbery!)
- But you can go in any direction but you have limited time (stamina)

!!! Important note !!!
Since the map is changing (we collect the coins), it must be part of the state => immutability is needed
"""


class CollectCoinsState:
    def __init__(self, map, i, j, stamina):
        self.map = map
        self.i = i
        self.j = j
        self.h = len(self.map)
        self.w = len(self.map[0])
        self.stamina = stamina

    def __eq__(self, other):
        return self.map == other.map and self.i == other.i and self.j == other.j and self.stamina == other.stamina

    def __hash__(self):
        map_hash = 0
        for row in self.map:
            for cell in row:
                map_hash ^= hash(cell)
        return map_hash ^ hash(self.i) ^ hash(self.j) ^ hash(self.stamina)

    def is_done(self):
        return self.stamina == 0

    def get_actions(self):
        actions = []
        if self.i < self.h - 1:
            actions.append((1, 0))
        if self.j < self.w - 1:
            actions.append((0, 1))
        if self.i > 0:
            actions.append((-1, 0))
        if self.j > 0:
            actions.append((0, -1))
        return actions

    def move(self, di, dj):
        i = self.i + di
        j = self.j + dj
        new_map = copy.deepcopy(self.map)
        new_map[i][j] = 0
        return CollectCoinsState(i=i, j=j, stamina=self.stamina - 1, map=new_map)


class CollectCoins:
    def __init__(self):
        self.state = self._init_state()

    def reset(self):
        self.state = self._init_state()

    def sample(self):
        pass    # TODO - not sure it is helpful if it just creates new states!

    def get_state(self):
        return self.state

    def get_actions(self):
        return self.state.get_actions()

    def is_done(self):
        return self.state.is_done()

    def step(self, action):
        di, dj = action
        new_state = self.state.move(di, dj)
        reward = self.state.map[new_state.i][new_state.j]
        self.state = new_state
        return reward

    @staticmethod
    def _init_state():
        return CollectCoinsState(i=0, j=0, stamina=7, map=[
            [0, 0, 0, 0],
            [-50, 1, 1, 0],
            [0, -50, 1, 5],
            [0, 0, 1, 100]
        ])


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
An agent that uses Q-learning (in fact, a variant called Tabular Q-learning)
"""


class QAgent:
    def __init__(self):
        self.actions = defaultdict(set)             # map state to possible actions at that state
        self.transitions = defaultdict(Counter)     # map tuple (state, action) to expected target states (with prob)
        self.rewards = defaultdict(float)           # map tuple (state, action, new_state) to reward
        self.q_values = defaultdict(float)          # map tuple (state, action) to expected value
        self.temperature = 1.                       # controls the number of random actions attempted
        self.discount = 0.9                         # discount factor used in Q-learning bellman update
        self.blending = 0.2

    def step(self, env) -> float:
        state = env.get_state()
        actions = env.get_actions()
        self.actions[state] = set(actions)          # TODO - useless to recompute every time!
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
        self.q_values[(state, action)] = \
            (1 - self.blending) * self.q_values[(state, action)] + \
            self.blending * (expected_reward + self.discount * expected_value)

    def temperature_decrease(self, decrease=0.1):
        self.temperature -= decrease
        self.temperature = max(self.temperature, 0.)

    def __str__(self):
        return str({'q_values': self.q_values, 'temperature': self.temperature})


"""
An agent that uses Deep Q-learning

!!! This is much more complex to do !!!
- you need to vectorize the obsevations (adapters for each game)
- you need to vectorize the actions (adapters for each game)
- ideally, you need to collect all possible actions (the DQN takes as input the state and outputs the action to perform)
- you need to collect a bunch of observation and sample a training minibatch from it
"""


class LinearEnvironmentVectorized(LinearEnvironment):
    def __init__(self):
        super().__init__()

    def vectorize(self, state, action):
        if action == 1:
            return torch.Tensor([state, 0, 1])
        elif action == 2:
            return torch.Tensor([state, 1, 0])


class DeepQAgent:
    def __init__(self):
        self.actions = defaultdict(set)                     # map state to possible actions at that state
        self.model = self._init_model()                     # the model used to make prediction
        self.learning_model = copy.deepcopy(self.model)     # the model to train
        self.transitions = []                               # to sample mini-batch from
        self.temperature = 1.                               # controls the number of random actions attempted
        self.discount = 0.9                                 # discount factor used in Q-learning bellman update
        self.blending = 0.2

    def step(self, env) -> float:
        state = env.get_state()
        actions = env.get_actions()
        self.actions[state] = actions
        action = self._select_best(env, state, actions)
        reward = env.step(action)
        self._value_iteration(env, state, action, env.get_state(), reward)
        return reward

    def _select_best(self, env, state, actions):
        """
        Selects the action with the best Q-value (expected long term reward)
        """
        if random.random() < self.temperature:
            return random.choice(actions)

        '''
        # Much slower than doing a bulk multiplication for all actions
        best_action = actions[0]
        best_action_reward = -1 * float('inf')
        for action in actions:
            reward = self.model(env.vectorize(state, action))
            if reward > best_action_reward:
                best_action = action
                best_action_reward = reward
        return best_action
        '''

        self.model.eval()
        xs = torch.cat([env.vectorize(state, action).unsqueeze(dim=0) for action in actions], dim=0)
        ys = self.model(xs)
        best_action_i = torch.argmax(ys, dim=0) # TODO - check dim?
        return actions[best_action_i]

    def _init_model(self):
        return nn.Sequential(
            nn.Linear(in_features=3, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1)
        )

    def _value_iteration(self, env, state, action, new_state, reward):
        self.transitions.append((state, action, new_state, reward))
        if len(self.transitions) >= 1000:
            self._train_model(env, self.transitions[:100])
            self.transitions.clear()

    def _train_model(self, env, mini_batch):
        ys = []
        xs = torch.cat([env.vectorize(state, action).unsqueeze(dim=0) for (state, action, _, _) in mini_batch], dim=0)
        for (_, _, new_state, reward) in mini_batch:
            new_actions = self.actions[new_state]
            if new_actions:
                new_xs = torch.cat([env.vectorize(new_state, action).unsqueeze(dim=0) for action in new_actions], dim=0)
                outputs = self.model(new_xs)
                expected_value, _ = torch.max(outputs, dim=0) # TODO - check dim?
                ys.append(reward + expected_value.item())
            else:
                ys.append(reward)
        ys = torch.Tensor(ys)

        loss_fct = nn.MSELoss()
        optimizer = optim.Adam(params=self.learning_model.parameters(), lr=1e-2)
        self.learning_model.train()
        outputs = self.learning_model(xs)
        loss = loss_fct(outputs, ys)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        self.model = copy.deepcopy(self.learning_model)

    def temperature_decrease(self, decrease=0.1):
        self.temperature -= decrease
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
    print("Random agent:", np.mean(rewards))


def train_q_agent(env, agent):
    rewards = []
    for epoch in range(1000):
        # first collect some stats
        '''
        for _ in range(10):
            env.sample()
            if not env.is_done():
                agent.step(env)
        '''

        # play a real game
        total_reward = 0.
        env.reset()
        while not env.is_done():
            total_reward += agent.step(env)
        rewards.append(total_reward)
        if (1 + epoch) % 100 == 0:
            print("Epoch", epoch + 1, ":", np.mean(rewards), " (temperature " + str(agent.temperature) + ")")
            agent.temperature_decrease(0.25)
            rewards.clear()


test_random_agent(env=LinearEnvironment())
print("-" * 20)
train_q_agent(env=LinearEnvironment(), agent=QAgent())
print("-" * 20)
train_q_agent(env=LinearEnvironmentVectorized(), agent=DeepQAgent())

print("-" * 20)

test_random_agent(env=FindYourWayEnv())
print("-" * 20)
train_q_agent(env=FindYourWayEnv(), agent=QAgent())

print("-" * 20)

test_random_agent(env=SlipperyFindYourWayEnv(slip_prob=0.2))
print("-" * 20)
train_q_agent(env=SlipperyFindYourWayEnv(slip_prob=0.2), agent=QAgent())

print("-" * 20)

test_random_agent(env=CollectCoins())
print("-" * 20)
train_q_agent(env=CollectCoins(), agent=QAgent())

