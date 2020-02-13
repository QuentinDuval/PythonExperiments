from collections import *
import itertools
import math
import sys
import time
from dataclasses import *
from typing import *

import numpy as np


"""
------------------------------------------------------------------------------------------------------------------------
UTILS
------------------------------------------------------------------------------------------------------------------------
"""


def debug(*args):
    print(*args, file=sys.stderr)


class Chronometer:
    def __init__(self):
        self.start_time = 0

    def start(self):
        self.start_time = time.time_ns()

    def spent(self):
        current = time.time_ns()
        return self._to_ms(current - self.start_time)

    @staticmethod
    def _to_ms(delay):
        return delay / 1_000_000


"""
------------------------------------------------------------------------------------------------------------------------
GAME ENTITIES
------------------------------------------------------------------------------------------------------------------------
"""


Distance = int
EntityId = int
Topology = Dict[EntityId, Dict[EntityId, Distance]]


@dataclass()
class Factory:
    owner: int
    cyborg_count: int
    production: int


@dataclass()
class Troop:
    owner: int
    source: EntityId
    destination: EntityId
    cyborg_count: int
    distance: Distance


@dataclass()
class GameState:
    factories: Dict[EntityId, Factory] = field(default_factory=dict)
    troops: Dict[EntityId, Troop] = field(default_factory=dict)

    @classmethod
    def read(cls):
        state = cls()
        entity_count = int(input())
        for i in range(entity_count):
            entity_id, entity_type, arg_1, arg_2, arg_3, arg_4, arg_5 = input().split()
            entity_id = int(entity_id)
            arg_1 = int(arg_1)
            arg_2 = int(arg_2)
            arg_3 = int(arg_3)
            arg_4 = int(arg_4)
            arg_5 = int(arg_5)
            if entity_type == "FACTORY":
                state.factories[entity_id] = Factory(owner=arg_1, cyborg_count=arg_2, production=arg_3)
            elif entity_type == "TROOP":
                state.troops[entity_id] = Troop(
                    owner=arg_1, source=arg_2, destination=arg_3,
                    cyborg_count=arg_4, distance=arg_5)
        return state


@dataclass()
class Wait:
    def __repr__(self):
        return "WAIT"


@dataclass()
class Move:
    source: EntityId
    destination: EntityId
    cyborg_count: int

    def __repr__(self):
        return "MOVE " + str(self.source) + " " + str(self.destination) + " " + str(self.cyborg_count)


Action = Union[Wait, Move]


"""
------------------------------------------------------------------------------------------------------------------------
AGENT
------------------------------------------------------------------------------------------------------------------------
"""


class Agent:
    def __init__(self):
        self.chrono = Chronometer()

    def get_action(self, topology: Topology, game_state: GameState) -> Action:
        for f_id, f in game_state.factories.items():
            if f.owner > 0 and f.cyborg_count > 10:
                return self.send_troop_from(f_id, topology, game_state)
        return Wait()

    def send_troop_from(self, source: EntityId, topology: Topology, game_state: GameState) -> Action:
        for f_id, f in game_state.factories.items():
            if f.owner == 0:
                return Move(source, f_id, game_state.factories[source].cyborg_count - 10)
        for f_id, f in game_state.factories.items():
            if f.owner == -1:
                return Move(source, f_id, game_state.factories[source].cyborg_count - 10)
        return Wait()


"""
------------------------------------------------------------------------------------------------------------------------
GAME LOOP
------------------------------------------------------------------------------------------------------------------------
"""


def read_topology() -> Topology:
    topology = defaultdict(lambda: defaultdict(Distance))
    factory_count = int(input())
    link_count = int(input())
    for i in range(link_count):
        factory_1, factory_2, distance = [int(j) for j in input().split()]
        topology[factory_1][factory_2] = distance
        topology[factory_2][factory_1] = distance
    return topology


def game_loop():
    agent = Agent()
    topology = read_topology()
    while True:
        game_state = GameState.read()
        action = agent.get_action(topology, game_state)
        print(action)


if __name__ == '__main__':
    game_loop()
