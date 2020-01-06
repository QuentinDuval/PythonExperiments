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
Utilities for debugging an Geometry
"""


def debug(*args):
    print(*args, file=sys.stderr)


T = TypeVar('T')


"""
Geometry
"""


Angle = float
Duration = float
Mass = float
Vector = np.ndarray


def vector(x: int, y: int) -> Vector:
    return np.array([x, y])


def norm2(v) -> float:
    return np.dot(v, v)


def norm(v) -> float:
    return math.sqrt(norm2(v))


def distance2(v1: Vector, v2: Vector) -> float:
    v = v1 - v2
    return np.dot(v, v)


def distance(v1: Vector, v2: Vector) -> float:
    return math.sqrt(distance2(v1, v2))


def get_angle(v: Vector) -> Angle:
    # Get angle from a vector (x, y) in radian
    x, y = v
    if x > 0:
        return np.arctan(y / x)
    if x < 0:
        return math.pi - np.arctan(- y / x)
    return math.pi / 2 if y >= 0 else -math.pi / 2


def mod_angle(angle: Angle) -> Angle:
    if angle > 2 * math.pi:
        return angle - 2 * math.pi
    if angle < 0:
        return angle + 2 * math.pi
    return angle


"""
------------------------------------------------------------------------------------------------------------------------
GAME CONSTANTS
------------------------------------------------------------------------------------------------------------------------
"""


WIDTH = 16001
HEIGHT = 7501

MAX_THRUST = 150
MAX_THROW_POWER = 500

WIZARD_RADIUS = 400
BLUDGER_RADIUS = 200
SNAFFLE_RADIUS = 150

FRICTION_SNAFFLE = 0.75
FRICTION_WIZARD = 0.75
FRICTION_BLUDGER = 0.9

MASS_WIZARD = 1.0
MASS_SNAFFLE = 0.5
MASS_BLUDGER = 9.0


GOAL_Y_LO = 2150
GOAL_Y_HI = 5500
GOAL_Y_CENTER = (GOAL_Y_HI + GOAL_Y_LO) // 2


class Goal(NamedTuple):
    x: int
    y_lo: int
    y_hi: int
    center: Vector

    @classmethod
    def create(cls, x):
        return cls(x=x, y_lo=GOAL_Y_LO, y_hi=GOAL_Y_HI, center=vector(x, GOAL_Y_CENTER))


LEFT_GOAL: Goal = Goal.create(x=0)
RIGHT_GOAL: Goal = Goal.create(x=WIDTH)
OWN_GOALS = (LEFT_GOAL, RIGHT_GOAL)


MANA_COST_OBLIVIATE = 5
MANA_COST_PETRIFICUS = 10
MANA_COST_ACCIO = 15
MANA_COST_FLIPENDO = 20

DURATION_OBLIVIATE = 4
DURATION_PETRIFICUS = 1
DURATION_ACCIO = 6
DURATION_FLIPPENDO = 3


"""
------------------------------------------------------------------------------------------------------------------------
MAIN DATA STRUCTURES
------------------------------------------------------------------------------------------------------------------------
"""


@dataclass(frozen=False)
class PlayerStatus:
    score: int
    magic: int


def read_status():
    score, magic = [int(i) for i in input().split()]
    return PlayerStatus(score=score, magic=magic)


EntityId = int


class Wizard(NamedTuple):
    id: EntityId
    position: Vector
    speed: Vector
    has_snaffle: bool

    def __eq__(self, other):
        return self.id == other.id\
            and np.array_equal(self.position, other.position)\
            and np.array_equal(self.speed, other.speed)\
            and self.has_snaffle == other.has_snaffle


class Snaffle(NamedTuple):
    id: EntityId
    position: Vector
    speed: Vector

    def __eq__(self, other):
        return self.id == other.id\
            and np.array_equal(self.position, other.position)\
            and np.array_equal(self.speed, other.speed)


class Bludger(NamedTuple):
    id: EntityId
    position: Vector
    speed: Vector

    def __eq__(self, other):
        return self.id == other.id\
            and np.array_equal(self.position, other.position)\
            and np.array_equal(self.speed, other.speed)


def find_by_id(entities: List[T], identity: EntityId) -> T:
    for entity in entities:
        if entity.id == identity:
            return entity
    return None


def assign_closest_snaffles(wizards: List[Wizard], all_snaffles: List[Snaffle]) -> Dict[EntityId, Snaffle]:
    def try_assignment_order(ordered_wizards: List[Wizard]) -> Tuple[Dict[EntityId, Snaffle], float]:
        assignments = dict()
        sum_distances = 0.
        snaffles = list(all_snaffles)
        for wizard in ordered_wizards:
            snaffle = min(snaffles, key=lambda s: distance2(wizard.position, s.position), default=None)
            if snaffle is not None:
                assignments[wizard.id] = snaffle
                snaffles.remove(snaffle)
                sum_distances += distance2(wizard.position, snaffle.position)
        return assignments, sum_distances

    assignment_1, cost_1 = try_assignment_order(wizards)
    assignment_2, cost_2 = try_assignment_order(wizards[::-1])
    return assignment_1 if cost_1 <= cost_2 else assignment_2


"""
------------------------------------------------------------------------------------------------------------------------
GAME STATE
------------------------------------------------------------------------------------------------------------------------
"""


@dataclass(frozen=True)
class GameState:
    player_status: PlayerStatus
    opponent_status: PlayerStatus
    player_goal: Goal
    opponent_goal: Goal
    player_wizards: List[Wizard] = field(default_factory=list)
    opponent_wizards: List[Wizard] = field(default_factory=list)
    snaffles: List[Snaffle] = field(default_factory=list)
    bludgers: List[Bludger] = field(default_factory=list)

    def init_next(self):
        return GameState(
            player_status=self.player_status, opponent_status=self.opponent_status,
            player_goal=self.player_goal, opponent_goal=self.opponent_goal,
            player_wizards=[], opponent_wizards=[],
            snaffles=[], bludgers=[])


def read_state(player_status, opponent_status, player_goal: Goal, opponent_goal: Goal) -> GameState:
    debug("read state")
    game_state = GameState(player_status, opponent_status, player_goal, opponent_goal)
    entity_nb = int(input())
    for _ in range(entity_nb):
        entity_id, entity_type, x, y, vx, vy, state = input().split()
        entity_id = int(entity_id)
        position = vector(int(x), int(y))
        speed = vector(int(vx), int(vy))
        has_snaffle = int(state) > 0
        if entity_type == "WIZARD":
            game_state.player_wizards.append(Wizard(id=entity_id, position=position, speed=speed, has_snaffle=has_snaffle))
        elif entity_type == "OPPONENT_WIZARD":
            game_state.opponent_wizards.append(Wizard(id=entity_id, position=position, speed=speed, has_snaffle=has_snaffle))
        elif entity_type == "BLUDGER":
            game_state.bludgers.append(Bludger(id=entity_id, position=position, speed=speed))
        elif entity_type == "SNAFFLE":
            game_state.snaffles.append(Snaffle(id=entity_id, position=position, speed=speed))
    return game_state


"""
------------------------------------------------------------------------------------------------------------------------
ACTIONS
------------------------------------------------------------------------------------------------------------------------
"""


class Move(NamedTuple):
    is_throw: bool  # Otherwise, it is a move
    direction: Vector
    power: int

    def __repr__(self):
        action_type = "THROW" if self.is_throw else "MOVE"
        x, y = self.direction
        return action_type + " " + str(x) + " " + str(y) + " " + str(self.power)


class Spell(NamedTuple):
    name: str
    target_id: int

    def __repr__(self):
        return self.name + " " + str(self.target_id)


Action = Union[Move, Spell]


def accio_power(wizard: Wizard, snaffle: Snaffle) -> float:
    dist2 = distance2(wizard.position, snaffle.position)
    return min(3000/(dist2/1_000_000), 1000)


def flippendo_power(wizard: Wizard, snaffle: Snaffle) -> float:
    dist2 = distance2(wizard.position, snaffle.position)
    return min(6000/(dist2/1_000_000), 1000)


"""
------------------------------------------------------------------------------------------------------------------------
PHYSIC ENGINE
------------------------------------------------------------------------------------------------------------------------
"""


def intersect_vertical_line(pos1: Vector, pos2: Vector, x_line: int) -> Optional[Tuple[int, Duration]]:
    # Quick check: should be on each side of the goal
    dx1 = pos1[0] - x_line
    dx2 = pos2[0] - x_line
    if dx1 * dx2 > 0.:
        return None

    # Find the intersection point
    x1, y1 = pos1
    dx, dy = pos2 - pos1
    dt = (x_line - x1) / dx     # Solve x1 + dx * dt = x_line for dt
    y_cross = y1 + dy * dt      # Then move y to the intersection point
    return int(y_cross), dt


def intersect_horizontal_line(pos1: Vector, pos2: Vector, y_line: int) -> Optional[Tuple[int, Duration]]:
    # Quick check: should be on each side of the goal
    dy1 = pos1[1] - y_line
    dy2 = pos2[1] - y_line
    if dy1 * dy2 > 0.:
        return None

    # Find the intersection point
    x1, y1 = pos1
    dx, dy = pos2 - pos1
    dt = (y_line - y1) / dy     # Solve y1 + dy * dt = y_line for dt
    x_cross = x1 + dx * dt      # Then move x to the intersection point
    return int(x_cross), dt


def apply_force(entity: T, thrust: float, destination: Vector, friction: float, mass: Mass, dt=1.0) -> T:
    # TODO - there is a bug here... the prev_snaffle, curr_snaffle will not be the correct line!

    # Update the speed vector
    position = entity.position
    if thrust > 0.:
        direction = get_angle(destination - position)
        force = np.array([thrust * math.cos(direction), thrust * math.sin(direction)])
        dv_dt = force / mass
    else:
        dv_dt = np.zeros(shape=(2,))
    new_speed = entity.speed + dv_dt * dt

    # Compute the new position (assumption: there can be at most one collision with the walls)
    new_position = np.round(position + new_speed * dt)
    for y_line in 0, HEIGHT:
        intersection = intersect_horizontal_line(position, new_position, y_line)
        if intersection is not None:
            x_cross, dt_till_hit = intersection
            position = vector(x_cross, y_line)
            new_speed[1] *= -1
            new_position = np.round(position + new_speed * (dt - dt_till_hit))

    # TODO - take into account the X borders ? beware for snaffles, they need to cross the goal
    return entity._replace(
        position=new_position,
        speed=np.trunc(new_speed * friction))


def intersect_goal(snaffle1: Snaffle, snaffle2: Snaffle, goal: Goal) -> bool:
    intersection = intersect_vertical_line(snaffle1.position, snaffle2.position, goal.x)
    if intersection is None:
        return False

    y_cross, dt = intersection
    return goal.y_lo <= y_cross <= goal.y_hi


'''
def simulate(state: GameState, actions: List[Tuple[Wizard, Action]]) -> GameState:
    next_state = state.init_next()

    # TODO - take into account the last longing spells?

    # Move the snaffle
    for snaffle in state.snaffles:
        thrust = 0.
        destination = state.opponent_goal.center
        for action_wizard, action in actions:
            if np.array_equal(action_wizard.position, snaffle.position):
                if isinstance(action, Move) and action.is_throw:
                    thrust = action.power   # TODO - inherit the speed of the player?
                    destination = action.direction
                    break
        next_snaffle = apply_force(snaffle, thrust=thrust, destination=destination, friction=0.75, mass=0.5, dt=1.0)
        if intersect_goal(snaffle, next_snaffle, state.opponent_goal):
            next_state.player_status.score += 1
        elif intersect_goal(snaffle, next_snaffle, state.player_goal):
            next_state.opponent_status.score += 1
        else:
            next_state.snaffles.append(next_snaffle)

    # Move the wizard
    for wizard in state.player_wizards:
        thrust = 0.
        destination = state.opponent_goal.center
        for action_wizard, action in actions:
            if action_wizard.id == wizard.id:
                if isinstance(action, Move) and not action.is_throw:
                    thrust = action.power
                    destination = action.direction
                    break
        # TODO - detection of catching / throwing a snaffle
        next_wizard = apply_force(wizard, thrust=thrust, destination=destination, friction=0.75, mass=1.0, dt=1.0)
        next_state.player_wizards.append(next_wizard)

    # Move the opponent wizard: TODO - move them according to a basic AI
    for wizard in state.opponent_wizards:
        # TODO - detection of catching / throwing a snaffle
        next_wizard = apply_force(wizard, thrust=0., destination=state.player_goal.center, friction=0.75, mass=1.0, dt=1.0)
        next_state.opponent_wizards.append(next_wizard)

    # Move the bludgers: TODO - how does the acceleration of bludgers work?
    for bludger in state.bludgers:
        next_bludger = apply_force(bludger, thrust=0., destination=bludger.position, friction=0.9, mass=8.0, dt=1.0)
        next_state.bludgers.append(next_bludger)

    # Increase the magic
    next_state.player_status.magic += 1
    next_state.opponent_status.magic += 1
    return next_state
'''


"""
------------------------------------------------------------------------------------------------------------------------
BASIC AGENTS
------------------------------------------------------------------------------------------------------------------------
"""


class Agent(abc.ABC):
    def on_turn_start(self, state: GameState):
        pass

    @abc.abstractmethod
    def get_actions(self, state: GameState) -> List[Move]:
        pass


class StupidAgent(Agent):
    def get_actions(self, state: GameState) -> List[Move]:
        return [Move(is_throw=False, direction=vector(8000, 3750), power=100),
                Move(is_throw=False, direction=vector(8000, 3750), power=100)]


class GrabClosestAndShootTowardGoal(Agent):
    """
    You need to keep a state here, in order to avoid the situation in which your agent changes
    direction suddenly and remains stuck, with two wizard each alternating snaffles

    TODO (IDEAS):
        - specialize the wizard: one to defend, one to attack (based on where is center of mass of balls, like soccer... move front):
            - the defender should try to intercept opponent (move between opponent and goal)
            - the defender should put the balls on the side (where opponent are not, or toward other wizard)
            - stick to one target for the attacking player?
        - use the detection of collisions to do something different (change target?)
    """

    def __init__(self):
        self.on_accio: Dict[EntityId, int] = {}
        self.on_flipendo: Dict[EntityId, int] = {}
        self.player_snaffles: Dict[EntityId, Wizard] = {}
        self.opponent_snaffles: Dict[EntityId, Wizard] = {}

    def on_turn_start(self, state: GameState):
        self._decrease_duration(self.on_accio)
        self._decrease_duration(self.on_flipendo)
        self.player_snaffles = self._find_snaffles_owned(state.snaffles, state.player_wizards)
        self.opponent_snaffles = self._find_snaffles_owned(state.snaffles, state.opponent_wizards)

    def get_actions(self, state: GameState) -> List[Action]:
        actions: List[Action] = [None] * len(state.player_wizards)

        # Book-keeping of which snaffles are owned or not
        free_snaffles = [s for s in state.snaffles
                         if s.id not in self.opponent_snaffles
                         if s.id not in self.player_snaffles]

        # Loop for opportunities to score a goal
        for i, wizard in enumerate(state.player_wizards):
            if wizard.has_snaffle:
                action = self._shoot_toward_goal(state, wizard, state.opponent_goal)
            else:
                action = self._try_flipendo(state, wizard, free_snaffles)
                if action is not None:
                    free_snaffles.remove(find_by_id(state.snaffles, action.target_id))
            actions[i] = action

        # Try to stop a ball that will go in my goal if it is possible (no one is near...)
        # TODO

        # Assign snaffles to remaining wizards with no actions
        free_wizards = [wizard for i, wizard in enumerate(state.player_wizards) if actions[i] is None]
        assignments = assign_closest_snaffles(free_wizards, free_snaffles)

        # Play with the snaffle that is assigned to you
        for i, wizard in enumerate(state.player_wizards):
            if actions[i] is None:
                actions[i] = self._capture_snaffle(state, wizard, assignments)
        return actions

    def _try_flipendo(self, state: GameState, wizard: Wizard, free_snaffles: List[Snaffle]) -> Spell:
        if state.player_status.magic < MANA_COST_FLIPENDO:
            return None

        for snaffle in free_snaffles:
            if snaffle.id not in self.on_flipendo:
                # TODO - maybe a bit restrictive to keep it just for goal
                if self._flipendo_to_goal(state, wizard, snaffle):
                    self.on_flipendo[snaffle.id] = DURATION_FLIPPENDO + 1
                    return Spell(name="FLIPENDO", target_id=snaffle.id)
        return None

    def _flipendo_to_goal(self, state: GameState, wizard: Wizard, snaffle: Snaffle) -> bool:
        if abs(snaffle.position[0] - state.opponent_goal.x) < WIDTH / 4:
            return False # Do not waste the flipendo

        # TODO - check there are no opponent in front
        # TODO - check for rebound (try the symetric of GOAL CENTER AT TOP AND LOW)
        goal = state.opponent_goal
        for _ in range(DURATION_FLIPPENDO):
            thrust = flippendo_power(wizard, snaffle)
            destination = snaffle.position + (snaffle.position - wizard.position)
            next_wizard = apply_force(wizard, thrust=0, destination=goal.center, friction=FRICTION_WIZARD, mass=MASS_WIZARD)
            next_snaffle = apply_force(snaffle, thrust=thrust, destination=destination, friction=FRICTION_SNAFFLE, mass=MASS_SNAFFLE, dt=1.0)
            if intersect_goal(snaffle, next_snaffle, state.opponent_goal):
                return True
            snaffle = next_snaffle
            wizard = next_wizard
        return False

    def _capture_snaffle(self, state: GameState, wizard: Wizard, assignments: Dict[EntityId, Snaffle]) -> Union[Move, Spell]:
        snaffle = assignments.get(wizard.id, None)
        if snaffle is not None:
            if self._can_accio(state, wizard, snaffle):
                self.on_accio[snaffle.id] = DURATION_ACCIO + 1
                return Spell(name="ACCIO", target_id=snaffle.id)
        else:
            snaffle = self._get_closest_snaffle_to_recapture(state, wizard)
            if snaffle is None:
                snaffle = state.snaffles[0]
        return Move(is_throw=False, direction=snaffle.position + snaffle.speed, power=MAX_THRUST)

    def _can_accio(self, state: GameState, wizard: Wizard, snaffle: Snaffle) -> bool:
        possible = state.player_status.magic >= MANA_COST_ACCIO
        possible &= snaffle.id not in self.on_accio
        possible &= distance2(snaffle.position, state.opponent_goal.center) > distance2(wizard.position, state.opponent_goal.center)
        possible &= distance2(wizard.position, snaffle.position) >= 3000 ** 2
        return possible

    def _get_closest_snaffle_to_recapture(self, state: GameState, wizard: Wizard):
        # In case there is but one snaffle and it is already taken, go toward the closest opponent that has a snaffle
        # TODO - try to use petrificus here
        min_dist = float('inf')
        closest_snaffle_id = None
        for snaffle_id, opponent_wizard in self.opponent_snaffles.items():
            dist = distance2(wizard.position, opponent_wizard.position)
            if dist < min_dist:
                min_dist = dist
                closest_snaffle_id = snaffle_id
        if closest_snaffle_id is not None:
            return find_by_id(state.snaffles, closest_snaffle_id)
        return None

    def _shoot_toward_goal(self, state: GameState, wizard: Wizard, goal: Goal):
        # TODO - if the goal is near, just shoot toward it
        # TODO - otherwise, there are better things to do to avoid opponents
        # TODO - avoid to shoot toward an opponent wizard OR toward a bludger
        # TODO - simulate to see if it will end up in goal, store that computation, to switch target if it will
        return Move(is_throw=True, direction=goal.center - wizard.speed, power=MAX_THROW_POWER)

    @staticmethod
    def _decrease_duration(durations: Dict[EntityId, int]):
        for entity_id, cd in list(durations.items()):
            if cd == 1:
                del durations[entity_id]
            else:
                durations[entity_id] = cd - 1

    @staticmethod
    def _find_snaffles_owned(snaffles: List[Snaffle], wizards: List[Wizard]) -> Dict[EntityId, Wizard]:
        owned = dict()
        for wizard in wizards:
            if wizard.has_snaffle:
                for snaffle in snaffles:
                    if distance2(snaffle.position, wizard.position) < WIZARD_RADIUS ** 2:
                        owned[snaffle.id] = wizard
                        break
        return owned


# TODO - an evaluation function that counts the goal + tries to put the balls in the adversary camp


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
        game_state = read_state(player_status, opponent_status, player_goal, opponent_goal)

        debug("player status:", player_status)
        debug("opponent status:", opponent_status)
        debug("game state:", game_state)

        agent.on_turn_start(game_state)
        for action in agent.get_actions(game_state):
            print(action)


if __name__ == '__main__':
    game_loop(agent=GrabClosestAndShootTowardGoal())
