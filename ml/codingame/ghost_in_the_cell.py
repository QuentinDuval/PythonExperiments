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


MAX_PRODUCTION = 3


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
Actions = List[Action]


"""
------------------------------------------------------------------------------------------------------------------------
AGENT
------------------------------------------------------------------------------------------------------------------------
"""


class Agent:
    MIN_TROOPS = 5

    def __init__(self):
        self.chrono = Chronometer()

    def get_action(self, topology: Topology, game_state: GameState) -> Actions:
        actions: Actions = []
        # TODO - apply the effect of on-going actions
        for f_id, f in game_state.factories.items():
            if f.owner > 0 and f.cyborg_count > self.MIN_TROOPS:
                actions.extend(self.send_troop_from(f_id, topology, game_state))
        if not actions:
            actions.append(Wait())
        return actions

    def send_troop_from(self, source: EntityId, topology: Topology, game_state: GameState) -> Actions:

        # TODO - bad formula for attractiveness? tries too often to assault the opponent base at start and lose
        # TODO - go through intermediary bases
        def attractiveness(f_id: int):
            f = game_state.factories[f_id]
            return topology[source][f_id] / (f.production + f.cyborg_count + 1)

        moves: Actions = []

        neutral = []
        for f_id, f in game_state.factories.items():
            if f_id != source and (f.owner != 1 or (f.owner == 1 and f.cyborg_count < self.MIN_TROOPS)):
                neutral.append(f_id)
        neutral.sort(key=attractiveness)

        excess_cyborgs = game_state.factories[source].cyborg_count - self.MIN_TROOPS
        for f_id in neutral:
            if not excess_cyborgs:
                return moves

            troops = min(excess_cyborgs, game_state.factories[f_id].cyborg_count + 1)
            moves.append(Move(source, f_id, troops))
            excess_cyborgs -= troops
        return moves


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
        actions = agent.get_action(topology, game_state)
        print(";".join(str(a) for a in actions))


if __name__ == '__main__':
    game_loop()
