from collections import *
import itertools
import heapq
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
INF_DISTANCE = 1_000

Distance = int
EntityId = int
Owner = int


@dataclass()
class Factory:
    owner: Owner
    cyborg_count: int
    production: int
    projected_count: int = 0


@dataclass()
class Troop:
    owner: Owner
    source: EntityId
    destination: EntityId
    cyborg_count: int
    distance: Distance


@dataclass()
class Bomb:
    owner: Owner
    source: EntityId
    destination: EntityId
    distance: Distance


@dataclass()
class GameState:
    turn_nb: int
    factories: Dict[EntityId, Factory] = field(default_factory=dict)
    troops: Dict[EntityId, Troop] = field(default_factory=dict)
    bombs: Dict[EntityId, Bomb] = field(default_factory=dict)

    @classmethod
    def read(cls, turn_nb: int):
        state = cls(turn_nb=turn_nb)
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


class Topology:
    def __init__(self):
        self.graph: Dict[EntityId, Dict[EntityId, Distance]] = defaultdict(dict)
        self.paths: Dict[EntityId, Dict[EntityId, EntityId]] = defaultdict(dict)
        self.camps: Dict[EntityId, Owner] = {}

    def distance(self, source: EntityId, destination: EntityId) -> Distance:
        return self.graph[source][destination]

    def next_move_hop(self, source: EntityId, destination: EntityId) -> EntityId:
        return self.paths[source][destination]

    def get_camp(self, factory_id: EntityId) -> Owner:
        return self.camps[factory_id]

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
        return topology

    @classmethod
    def from_graph(cls, graph):
        topology = Topology()
        for k, v in graph.items():
            topology.graph[k].update(v)
        return topology

    @classmethod
    def from_edges(cls, edges: List[Tuple[EntityId, EntityId, Distance]]):
        topology = Topology()
        for e1, e2, d in edges:
            topology.graph[e1][e2] = d
            topology.graph[e2][e1] = d
        return topology

    def compute_paths(self, game_state: GameState):
        # TODO: avoid "next hops" on a factory that has 0 production?
        graph = self.graph
        self.paths.clear()
        nodes = list(graph.keys())
        modified_dist: Dict[EntityId, Dict[EntityId, float]] = defaultdict(dict)
        for s in nodes:
            self.paths[s][s] = 0
            for d in nodes:
                if s != d:
                    modified_dist[s][d] = graph[s].get(d, INF_DISTANCE) ** 2
                    self.paths[s][d] = d
        for k in range(len(nodes)):
            if game_state.factories[k].owner == 1:
                for s in nodes:
                    for d in nodes:
                        if s != d and s != k and d != k:
                            if modified_dist[s][d] > modified_dist[s][k] + modified_dist[k][d]:
                                modified_dist[s][d] = modified_dist[s][k] + modified_dist[k][d]
                                self.paths[s][d] = self.paths[s][k]

    def compute_camps(self, game_state: GameState):
        self.camps.clear()
        camp_dist = {}

        q = []
        for f_id, f in game_state.factories.items():
            if f.owner != 0:
                heapq.heappush(q, (0, f_id, f.owner))
                self.camps[f_id] = f.owner
                camp_dist[f_id] = 0
        while q:
            f_dist, f_id, f_owner = heapq.heappop(q)

            # Assign camp and take into account equalities
            if f_id not in camp_dist:
                self.camps[f_id] = f_owner
                camp_dist[f_id] = f_dist
            elif camp_dist[f_id] == f_dist and self.camps[f_id] != f_owner:
                self.camps[f_id] = 0

            # Visit the neighbors
            for neigh, dist in self.graph[f_id].items():
                if neigh not in self.camps:
                    heapq.heappush(q, (f_dist + dist, neigh, f_owner))


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
    BOMB_TURN = 2

    def __init__(self):
        self.chrono = Chronometer()
        # TODO - make the AI subjective to a player

    def get_action(self, topology: Topology, game_state: GameState) -> Actions:
        actions: Actions = []

        # TODO: need a strategic analysis globally?
        #   * how can you manage the increase otherwise?

        # TODO: bad start
        #   * if a factory is too occupied at first, you might just wait to more easily get it
        #   * currently, I get the factory, and then it is taken back right after... I just clear the mobs...

        # TODO: opening book, send a bomb + a troop directly afterwards (on the opponent base?)
        #   * to do this, you need to take into account remaining troops (in order not to send them twice)

        # TODO: anticipate your own move:
        #   * no need to send twice the reinforcements

        # TODO - !production is affected by bombs, differentiate between temporary down and no prod (in attractiveness)
        # TODO - increase action

        self.aggregate_current_moves(game_state)
        actions.extend(self.decide_movements(topology, game_state))
        if game_state.turn_nb == self.BOMB_TURN:
            actions.extend(self.send_bombs(topology, game_state))
        return [Wait()] if not actions else actions

    def aggregate_current_moves(self, game_state: GameState):
        # TODO - instead, try to find when the factory change allegance in the few turns ahead (simulation)
        for f in game_state.factories.values():
            f.projected_count = f.cyborg_count + f.production # + f.production * self.MAX_PROJ_TURN
        for t in game_state.troops.values():
            f = game_state.factories[t.destination]
            if f.owner == t.owner:
                f.projected_count += t.cyborg_count
            else:
                f.projected_count -= t.cyborg_count

    def decide_movements(self, topology: Topology, game_state: GameState) -> Actions:
        # TODO - you should also avoid you own bombs
        bomb_impacts = {}
        for b in game_state.bombs.values():
            if b.owner == -1:
                bomb_impacts[b.destination] = min(b.distance, bomb_impacts.get(b.destination, INF_DISTANCE))
        debug(bomb_impacts)

        actions = []
        for f_id, f in game_state.factories.items():
            if f.owner == 1:
                imminent_impact = bomb_impacts.get(f_id, INF_DISTANCE) <= 2
                if imminent_impact:
                    excess_cyborgs = game_state.factories[f_id].cyborg_count
                else:
                    # TODO - min number of troops should be function of the production too
                    min_troops = self.MIN_TROOPS
                    excess_cyborgs = game_state.factories[f_id].projected_count - min_troops
                debug(f_id, imminent_impact, excess_cyborgs)
                if excess_cyborgs > 0:
                    actions.extend(self.send_troop_from(f_id, topology, game_state, excess_cyborgs, bomb_impacts))
        return actions

    def send_troop_from(self, source: EntityId, topology: Topology,
                        game_state: GameState, excess_cyborgs: int,
                        bomb_impacts: Dict[EntityId, Distance]) -> Actions:

        def attractiveness(f_id: int):
            dist = topology.distance(source, f_id) # TODO - take into account dist with hops
            if dist < bomb_impacts.get(f_id, -INF_DISTANCE):
                return float('inf')
            f = game_state.factories[f_id]
            camp_factor = 1 if topology.get_camp(f_id) == 1 else 5  # More attractiveness for my camp
            return camp_factor * (f.projected_count + 1) * dist / (f.production ** 2 + 1)

        targets = []
        for f_id, f in game_state.factories.items():
            if f_id != source and (f.owner != 1 or (f.owner == 1 and f.projected_count < self.MIN_TROOPS)):
                targets.append(f_id)
        targets.sort(key=attractiveness)
        if not targets:
            return []

        moves: Actions = []
        for f_id in targets:
            if excess_cyborgs <= 0:
                return moves

            target = game_state.factories[f_id]
            if target.owner == 1:
                troops = min(excess_cyborgs, self.MIN_TROOPS - game_state.factories[f_id].projected_count)
            else:
                troops = min(excess_cyborgs, game_state.factories[f_id].projected_count + 1)

            if troops > 0:
                # debug("FROM", source, "- target", f_id, "with troops", troops)
                next_hop = topology.next_move_hop(source, f_id)
                if next_hop not in bomb_impacts:
                    # TODO - the bomb impacts should be in fact part of the shortest paths
                    moves.append(Move(source, next_hop, troops))
                else:
                    moves.append(Move(source, f_id, troops))
                excess_cyborgs -= troops
        if excess_cyborgs:
            next_hop = topology.next_move_hop(source, targets[0])
            moves.append(Move(source, next_hop, excess_cyborgs // 2))
        return moves

    def send_bombs(self, topology: Topology, game_state: GameState) -> Actions:
        # TODO - sometimes helps the opponent by eliminating neutral at far regions from his base, helping him
        targets = []
        for f_id, f in game_state.factories.items():
            if topology.get_camp(f_id) == -1:
                targets.append(f_id)
        targets.sort(key=lambda t_id: game_state.factories[t_id].production, reverse=True)

        sources = []
        for f_id, f in game_state.factories.items():
            if f.owner == 1:
                sources.append(f_id)

        actions = []
        for i in range(MAX_BOMBS):
            target = targets[i]
            source = min(sources, key=lambda s_id: topology.distance(s_id, target))
            actions.append(SendBomb(source, target))
        return actions


"""
------------------------------------------------------------------------------------------------------------------------
GAME LOOP
------------------------------------------------------------------------------------------------------------------------
"""


def identify_bomb_target(topology: Topology, game_state: GameState, bomb: Bomb):
    # For opponent bombs, we do not get the distance nor the destination
    closest = -1
    closest_distance = 0
    closest_production = -1
    for f_id, distance in topology.graph[bomb.source].items():
        factory = game_state.factories[f_id]
        if factory.owner == 1 and factory.production > closest_production:
            closest = f_id
            closest_distance = distance - 1
            closest_production = factory.production
    bomb.destination = closest
    bomb.distance = closest_distance


def enrich(memory: GameState, game_state: GameState, topology: Topology):
    for b_id, bomb in game_state.bombs.items():
        if bomb.owner == -1:
            if b_id not in memory.bombs:
                identify_bomb_target(topology, game_state, bomb)
                debug(bomb)
            else:
                bomb.destination = memory.bombs[b_id].destination
                bomb.distance = memory.bombs[b_id].distance - 1


def game_loop():
    agent = Agent()
    topology = Topology.read()
    debug(topology)

    memory: GameState = None
    for turn_nb in itertools.count():
        game_state = GameState.read(turn_nb)
        if turn_nb == 0:
            topology.compute_camps(game_state)
        topology.compute_paths(game_state)
        enrich(memory, game_state, topology)
        actions = agent.get_action(topology, game_state)
        memory = game_state
        print(";".join(str(a) for a in actions))


if __name__ == '__main__':
    game_loop()
