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
MAX_BOMBS = 2

Distance = int
EntityId = int


class Topology:
    def __init__(self):
        self.graph: Dict[EntityId, Dict[EntityId, Distance]] = defaultdict(dict)
        self.paths: Dict[EntityId, Dict[EntityId, EntityId]] = {}

    def distance(self, source: EntityId, destination: EntityId) -> Distance:
        return self.graph[source][destination]

    def next_move_hop(self, source: EntityId, destination: EntityId) -> EntityId:
        return self.paths[source][destination]

    def __repr__(self):
        s = "GRAPH\n"
        s += str(self.graph)
        s += "\nPATHS\n"
        s += str(self.paths)
        return s

    @classmethod
    def read(cls):
        topology = Topology()
        factory_count = int(input())
        link_count = int(input())
        for _ in range(link_count):
            factory_1, factory_2, distance = [int(j) for j in input().split()]
            topology.graph[factory_1][factory_2] = distance
            topology.graph[factory_2][factory_1] = distance
        topology.paths = cls.compute_paths(topology.graph)
        return topology

    @classmethod
    def from_graph(cls, graph):
        topology = Topology()
        for k, v in graph.items():
            topology.graph[k].update(v)
        topology.paths = cls.compute_paths(topology.graph)
        return topology

    @staticmethod
    def compute_paths(graph):
        nodes = list(graph.keys())
        paths: Dict[EntityId, Dict[EntityId, EntityId]] = defaultdict(dict)
        modified_dist: Dict[EntityId, Dict[EntityId, Distance]] = defaultdict(dict)
        for s in nodes:
            paths[s][s] = 0
            for d in nodes:
                if s != d:
                    modified_dist[s][d] = graph[s][d] ** 2
                    paths[s][d] = d
        for k in range(len(nodes)):
            for s in nodes:
                for d in nodes:
                    if s != d and s != k and d != k:
                        if modified_dist[s][d] > modified_dist[s][k] + modified_dist[k][d]:
                            modified_dist[s][d] = modified_dist[s][k] + modified_dist[k][d]
                            paths[s][d] = k
        return paths


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
class Bomb:
    owner: int
    source: EntityId
    destination: EntityId
    distance: Distance


@dataclass()
class GameState:
    factories: Dict[EntityId, Factory] = field(default_factory=dict)
    troops: Dict[EntityId, Troop] = field(default_factory=dict)
    bombs: Dict[EntityId, Bomb] = field(default_factory=dict)

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
            elif entity_type == "BOMB":
                state.bombs[entity_id] = Bomb(owner=arg_1, source=arg_2, destination=arg_3, distance=arg_4)

        return state


@dataclass()
class Wait:
    def __repr__(self):
        return "WAIT"


@dataclass()
class SendBomb:
    source: EntityId
    destination: EntityId

    def __repr__(self):
        return "BOMB " + str(self.source) + " " + str(self.destination)


@dataclass()
class Increase:
    factory_id: EntityId

    def __repr__(self):
        return "INC " + str(self.factory_id)


@dataclass()
class Move:
    source: EntityId
    destination: EntityId
    cyborg_count: int

    def __repr__(self):
        return "MOVE " + str(self.source) + " " + str(self.destination) + " " + str(self.cyborg_count)


Action = Union[Wait, Move, SendBomb, Increase]
Actions = List[Action]


"""
------------------------------------------------------------------------------------------------------------------------
AGENT
------------------------------------------------------------------------------------------------------------------------
"""


class Agent:
    MIN_TROOPS = 1

    def __init__(self):
        self.chrono = Chronometer()
        self.bomb_sent = 0

    def get_action(self, topology: Topology, game_state: GameState) -> Actions:
        actions: Actions = []
        # TODO - increase action
        # TODO - pre-treat the topology to detect your camp and defend it
        # TODO - pre-treat the topology to pass through as many  nodes as possible
        # TODO - apply the effect of on-going actions
        for f_id, f in game_state.factories.items():
            if f.owner > 0 and f.cyborg_count > self.MIN_TROOPS:
                actions.extend(self.send_troop_from(f_id, topology, game_state))
        actions.extend(self.send_bombs(topology, game_state))
        if not actions:
            actions.append(Wait())
        return actions

    def send_bombs(self, topology: Topology, game_state: GameState) -> Actions:
        if self.bomb_sent >= MAX_BOMBS:
            return []

        targets = []
        for f_id, f in game_state.factories.items():
            if f.owner == -1:
                targets.append(f_id)
        targets.sort(key=lambda f_id: game_state.factories[f_id].production)
        if not targets:
            return []

        target = targets[-1]

        for b in game_state.bombs.values():
            if b.owner == 1 and b.destination == target:
                return []

        sources = []
        for f_id, f in game_state.factories.items():
            if f.owner == 1:
                sources.append(f_id)
        sources.sort(key=lambda f_id: topology.distance(f_id, target))
        if not sources:
            return []

        actions = [SendBomb(sources[0], target)]
        self.bomb_sent += 1
        return actions

    def send_troop_from(self, source: EntityId, topology: Topology, game_state: GameState) -> Actions:

        # TODO - bad formula for attractiveness? tries too often to assault the opponent base at start and lose
        # TODO - go through intermediary bases
        def attractiveness(f_id: int):
            f = game_state.factories[f_id]
            return topology.distance(source, f_id) / (f.production + f.cyborg_count + 1)

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
            next_hop = topology.next_move_hop(source, f_id)
            moves.append(Move(source, next_hop, troops))
            excess_cyborgs -= troops
        return moves


"""
------------------------------------------------------------------------------------------------------------------------
GAME LOOP
------------------------------------------------------------------------------------------------------------------------
"""


def game_loop():
    agent = Agent()
    topology = Topology.read()
    debug(topology)
    for turn_nb in itertools.count():
        game_state = GameState.read()
        # TODO - voronoi to identify camps at beginning (turn 0)
        actions = agent.get_action(topology, game_state)
        print(";".join(str(a) for a in actions))


if __name__ == '__main__':
    game_loop()
