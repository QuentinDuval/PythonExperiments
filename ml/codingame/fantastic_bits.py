"""
Grab Snaffles and try to throw them through the opponent's goal!
Move towards a Snaffle and use your team id to determine where you need to throw it.
"""


import abc
from collections import *
from dataclasses import *
import numpy as np
from typing import *
import sys
import math


"""
Utilities
"""


def debug(*args):
    print(*args, file=sys.stderr)


"""
Input acquisition & Game constants
"""


Vector = np.ndarray


def vector(x: int, y: int) -> Vector:
    return np.array([x, y])


WIDTH = 16001
HEIGHT = 7501


class Goal(NamedTuple):
    x: int
    y_lo: int
    y_hi: int


LEFT_GOAL = Goal(x=0, y_lo=2150, y_hi=5500)
RIGHT_GOAL = Goal(x=WIDTH, y_lo=2150, y_hi=5500)
OWN_GOALS = (LEFT_GOAL, RIGHT_GOAL)


@dataclass()
class PlayerStatus:
    score: int
    magic: int


def read_status():
    score, magic = [int(i) for i in input().split()]
    return PlayerStatus(score=score, magic=magic)


class Wizard(NamedTuple):
    id: int
    position: Vector
    speed: Vector
    has_snaffle: bool


class Snaffle(NamedTuple):
    id: int
    position: Vector
    speed: Vector


@dataclass(frozen=False)
class GameState:
    player_wizards: List[Wizard]
    opponent_wizards: List[Wizard]
    snaffles: List[Snaffle]

    @classmethod
    def empty(cls):
        return cls(player_wizards=[], opponent_wizards=[], snaffles=[])


def read_state() -> GameState:
    game_state = GameState.empty()
    entity_nb = int(input())
    for i in range(entity_nb):
        entity_id, entity_type, x, y, vx, vy, state = input().split()
        entity_id = int(entity_id)
        position = vector(int(x), int(y))
        speed = vector(int(vx), int(vy))
        has_snaffle = int(state) > 0
        if entity_type == "WIZARD":
            game_state.player_wizards.append(Wizard(id=entity_id, position=position, speed=speed, has_snaffle=has_snaffle))
        elif entity_type == "OPPONENT_WIZARD":
            game_state.opponent_wizards.append(Wizard(id=entity_id, position=position, speed=speed, has_snaffle=has_snaffle))
        else:
            game_state.snaffles.append(Snaffle(id=entity_id, position=position, speed=speed))
    return game_state


class Action(NamedTuple):
    # Action for each wizard (0 ≤ thrust ≤ 150, 0 ≤ power ≤ 500)
    # i.e.: "MOVE x y thrust" or "THROW x y power"

    is_throw: bool  # Otherwise, it is a move
    direction: Vector
    power: int

    def __repr__(self):
        action_type = "THROW" if self.is_throw else "MOVE"
        x, y = self.direction
        return action_type + " " + str(x) + " " + str(y) + " " + str(self.power)


"""
------------------------------------------------------------------------------------------------------------------------
AGENTS
------------------------------------------------------------------------------------------------------------------------
"""


class Agent(abc.ABC):
    @abc.abstractmethod
    def get_actions(self):
        pass


class StupidAgent(Agent):
    def get_actions(self):
        return [Action(is_throw=False, direction=vector(8000, 3750), power=100),
                Action(is_throw=False, direction=vector(8000, 3750), power=100)]



"""
------------------------------------------------------------------------------------------------------------------------
GAME LOOP
------------------------------------------------------------------------------------------------------------------------
"""


def game_loop(agent: Agent):
    # if 0 you need to score on the right of the map, if 1 you need to score on the left
    my_team_id = int(input())
    player_goal = OWN_GOALS[my_team_id]
    opponent_goal = OWN_GOALS[1-my_team_id]

    debug("my team id:", my_team_id)
    debug("my goal:", player_goal)
    debug("target goal:", opponent_goal)

    while True:
        player_status = read_status()
        opponent_status = read_status()
        game_state = read_state()

        debug("player status:", player_status)
        debug("opponent status:", opponent_status)
        debug("game state:", game_state)

        for action in agent.get_actions():
            print(action)


if __name__ == '__main__':
    game_loop(agent=StupidAgent())
