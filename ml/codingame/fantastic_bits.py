"""
Grab Snaffles and try to throw them through the opponent's goal!
Move towards a Snaffle and use your team id to determine where you need to throw it.
"""


import abc
from collections import *
from dataclasses import *
import enum
import numpy as np
import time
from typing import *
import sys
import math


"""
Utilities for debugging
"""


def debug(*args):
    print(*args, file=sys.stderr)


T = TypeVar('T')


class Chronometer:
    def __init__(self):
        self.start_time = 0

    def start(self):
        self.start_time = time.time_ns()

    def spent(self):
        current = time.time_ns()
        return self._to_ms(current - self.start_time)

    def _to_ms(self, delay):
        return delay / 1_000_000


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


GOAL_Y_LO = 2150 + 50   # Radius of goal post
GOAL_Y_HI = 5500 - 50   # Radius of goal post
GOAL_Y_CENTER = (GOAL_Y_HI + GOAL_Y_LO) // 2


class Goal(NamedTuple):
    x: int
    y_lo: int
    y_hi: int
    bottom: Vector
    center: Vector
    top: Vector

    @classmethod
    def create(cls, x):
        return cls(x=x, y_lo=GOAL_Y_LO, y_hi=GOAL_Y_HI,
                   bottom=vector(x, GOAL_Y_LO + 500),
                   top=vector(x, GOAL_Y_HI - 500),
                   center=vector(x, GOAL_Y_CENTER))


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
            player_wizards=[],
            opponent_wizards=[],
            snaffles=[],
            bludgers=[])


class FieldPart(enum.Enum):
    DEFENSE = 0
    MID_FIELD = 1
    ATTACK = 2


def get_field_part(state: GameState, entity: T) -> FieldPart:
    x = entity.position[0]
    goal = state.player_goal.x
    dx = abs(goal - x)
    if dx > 2 * WIDTH / 3:
        return FieldPart.ATTACK
    elif dx > WIDTH / 3:
        return FieldPart.MID_FIELD
    else:
        return FieldPart.DEFENSE


def read_state(player_status, opponent_status, player_goal: Goal, opponent_goal: Goal) -> GameState:
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

    @classmethod
    def throw_ball(cls, direction: Vector):
        return Move(is_throw=True, direction=direction, power=MAX_THROW_POWER)

    @classmethod
    def move_toward(cls, direction: Vector):
        return Move(is_throw=False, direction=direction, power=MAX_THRUST)

    def __repr__(self):
        action_type = "THROW" if self.is_throw else "MOVE"
        x, y = self.direction
        return action_type + " " + str(x) + " " + str(y) + " " + str(self.power)


class Spell(NamedTuple):
    name: str
    target_id: int

    @classmethod
    def throw_accio(cls, state: GameState, target_id: EntityId):
        state.player_status.magic -= MANA_COST_ACCIO
        return cls(name="ACCIO", target_id=target_id)

    @classmethod
    def throw_flipendo(cls, state: GameState, target_id: EntityId):
        state.player_status.magic -= MANA_COST_ACCIO
        return cls(name="FLIPENDO", target_id=target_id)

    @classmethod
    def throw_petrificus(cls, state: GameState, target_id: EntityId):
        state.player_status.magic -= MANA_COST_PETRIFICUS
        return cls(name="PETRIFICUS", target_id=target_id)

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


# TODO - ultimately, take two trajectories
def on_trajectory(pos1: Vector, pos2: Vector, element: Vector, radius: float) -> bool:
    # Quick check: outside of the segment
    if max(distance2(element, pos1), distance2(element, pos2)) > distance2(pos1, pos2):
        return False

    # TODO - Compute the normal?
    pass


def intersect_vertical_line(pos1: Vector, pos2: Vector, x_line: int) -> Optional[Tuple[int, Duration]]:
    # Quick check: should be on each side of the goal
    dx1 = pos1[0] - x_line
    dx2 = pos2[0] - x_line
    if dx1 * dx2 > 0.:
        return None

    # Find the intersection point
    x1, y1 = pos1
    dx, dy = pos2 - pos1
    if dx == 0.:
        return None

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
    if dy == 0.:
        return None

    dt = (y_line - y1) / dy     # Solve y1 + dy * dt = y_line for dt
    x_cross = x1 + dx * dt      # Then move x to the intersection point
    return int(x_cross), dt


def apply_force(entity: T, thrust: float, destination: Vector, friction: float, mass: Mass, dt=1.0) -> T:
    is_snaffle = isinstance(entity, Snaffle)

    # Update the speed vector
    position = entity.position
    if thrust > 0.:
        direction = get_angle(destination - position)
        force = np.array([thrust * math.cos(direction), thrust * math.sin(direction)])
        dv_dt = force / mass
    else:
        dv_dt = np.zeros(shape=(2,))
    new_speed = entity.speed + dv_dt * dt

    # Compute the new position
    while dt > 0.:
        collided = False
        new_position = np.round(position + new_speed * dt)

        # Collision with horizontal lines
        for y_line in 0, HEIGHT:
            intersection = intersect_horizontal_line(position, new_position, y_line)
            if intersection is not None:
                x_cross, dt_till_hit = intersection
                if dt_till_hit == 0.:   # Already processed
                    continue

                position = vector(x_cross, y_line)
                new_speed[1] *= -1
                dt -= dt_till_hit
                collided = True
                break
        if collided:
            continue

        # Collision with vertical lines
        for x_line in 0, WIDTH:
            intersection = intersect_vertical_line(position, new_position, x_line)
            if intersection is not None:
                y_cross, dt_till_hit = intersection
                if dt_till_hit == 0.:   # Already processed
                    continue

                if is_snaffle and GOAL_Y_LO + 200 <= y_cross <= GOAL_Y_HI - 200:    # 200 for Safe margins
                    return entity._replace(position=vector(x_line, GOAL_Y_CENTER), speed=vector(0, 0))
                position = vector(x_line, y_cross)
                new_speed[0] *= -1
                dt -= dt_till_hit
                collided = True
                break

        # Otherwise, when no collisions
        if not collided:
            position = new_position
            break

    # Return the position of the new entity
    return entity._replace(
        position=position,
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


def assign_closest_snaffles(state: GameState, wizards: List[Wizard], free_snaffles: List[Snaffle],
                            defensive_factor: float, offensive_factor: float) -> Dict[EntityId, Snaffle]:

    # A metric to minimize that will allow the wizard to pick the right balls
    def metric(w: Wizard, s: Snaffle) -> float:
        return distance2(w.position, s.position)\
               + defensive_factor * distance2(s.position, state.player_goal.center)\
               + offensive_factor * distance2(s.position, state.opponent_goal.center)

    # Find the assignment that minimize the metric
    def try_assignment_order(ordered_wizards: List[Wizard]) -> Tuple[Dict[EntityId, Snaffle], float]:
        assignments = dict()
        sum_distances = 0.
        snaffles = list(free_snaffles)
        for wizard in ordered_wizards:
            snaffle = min(snaffles, key=lambda s: metric(wizard, s), default=None)
            if snaffle is not None:
                assignments[wizard.id] = snaffle
                snaffles.remove(snaffle)
                sum_distances += metric(wizard, snaffle)
        return assignments, sum_distances

    assignment_1, cost_1 = try_assignment_order(wizards)
    assignment_2, cost_2 = try_assignment_order(wizards[::-1])
    return assignment_1 if cost_1 <= cost_2 else assignment_2


class GrabClosestAndShootTowardGoal(Agent):
    """
    You need to keep a state here, in order to avoid the situation in which your agent changes
    direction suddenly and remains stuck, with two wizard each alternating snaffles

    TODO (IDEAS):
        - TRY SEVERAL APPROACHES, then take the one with the best SCORE
        - specialize the wizard: one to defend, one to attack (based on where is center of mass of balls, like soccer... move front):
            - the defender should try to intercept opponent (move between opponent and goal)
            - the defender should put the balls on the side (where opponent are not, or toward other wizard)
            - stick to one target for the attacking player?
        - use the detection of collisions to do something different (change target?)
        - simulate to see if it a snaffle will end up in goal, and switch target if it will
    """

    def __init__(self):
        self.on_accio: Dict[EntityId, int] = {}
        self.on_flipendo: Dict[EntityId, int] = {}
        self.player_snaffles: Dict[EntityId, Wizard] = {}
        self.opponent_snaffles: Dict[EntityId, Wizard] = {}
        self.total_snaffles = 0
        self.player_remaining_goal = 10
        self.opponent_remaining_goal = 10

    def on_turn_start(self, state: GameState):
        self._decrease_duration(self.on_accio)
        self._decrease_duration(self.on_flipendo)
        self.player_snaffles = self._find_snaffles_owned(state.snaffles, state.player_wizards)
        self.opponent_snaffles = self._find_snaffles_owned(state.snaffles, state.opponent_wizards)
        self.total_snaffles = max(self.total_snaffles, len(state.snaffles))
        self.player_remaining_goal = self.total_snaffles - state.player_status.score
        self.opponent_remaining_goal = self.total_snaffles - state.opponent_status.score

    def get_actions(self, state: GameState) -> List[Action]:
        actions: List[Action] = [None] * len(state.player_wizards)

        # Book-keeping of which snaffles are owned or not
        free_snaffles = [s for s in state.snaffles
                         if s.id not in self.opponent_snaffles
                         if s.id not in self.player_snaffles]

        # Look for opportunities to score a goal / prevent a goal
        for i, wizard in enumerate(state.player_wizards):
            if wizard.has_snaffle:
                action = self._shoot_ball(state, wizard, free_snaffles)
            else:
                action = self._try_flipendo(state, wizard, free_snaffles)
                if action is not None:
                    free_snaffles.remove(find_by_id(state.snaffles, action.target_id))
            actions[i] = action

        # Look for opportunities to prevent a goal
        # TODO - petrificus

        # Assign snaffles to remaining wizards with no actions
        free_wizards = [wizard for i, wizard in enumerate(state.player_wizards) if actions[i] is None]
        assignments = assign_closest_snaffles(state, free_wizards, free_snaffles,
                                              defensive_factor=0.1,
                                              offensive_factor=0.0)
        # TODO - vary the defensive factor based on the progression of the game (keep the number of balls at beginning)

        # Play with the snaffle that is assigned to you
        for i, wizard in enumerate(state.player_wizards):
            if actions[i] is None:
                actions[i] = self._capture_snaffle(state, wizard, assignments)
        return actions

    def _try_flipendo(self, state: GameState, wizard: Wizard, free_snaffles: List[Snaffle]) -> Optional[Action]:
        if state.player_status.magic < MANA_COST_FLIPENDO:
            return None

        # TODO - try to move in a position to shoot if you are in the right half of the terrain

        for snaffle in free_snaffles:
            if snaffle.id not in self.on_flipendo:
                # TODO - maybe a bit restrictive to keep it just for goal (could be used to get rid of a ball)
                if self._flipendo_to_goal(state, wizard, snaffle):
                    self.on_flipendo[snaffle.id] = DURATION_FLIPPENDO + 1
                    return Spell.throw_flipendo(state, snaffle.id)
        return None

    def _flipendo_to_goal(self, state: GameState, wizard: Wizard, snaffle: Snaffle) -> bool:
        # Do not waste the flipendo if too close from the goal, except if it is to finish the game
        finish_it_mode = self.player_remaining_goal == 1 or self.opponent_remaining_goal == 1
        if finish_it_mode or distance2(snaffle.position, state.opponent_goal.center) < 3000 ** 2:
            return False

        # TODO - check there are no opponent in front
        # TODO - this logic is rather imprecise, and I get some misses when playing with rebounds
        future_wizard_target = min(state.snaffles, key=lambda s: distance2(s.position, wizard.position))
        for _ in range(DURATION_FLIPPENDO):
            thrust = flippendo_power(wizard, snaffle)
            destination = snaffle.position + (snaffle.position - wizard.position)
            next_wizard = apply_force(wizard, thrust=MAX_THRUST, destination=future_wizard_target.position, friction=FRICTION_WIZARD, mass=MASS_WIZARD)
            next_snaffle = apply_force(snaffle, thrust=thrust, destination=destination, friction=FRICTION_SNAFFLE, mass=MASS_SNAFFLE)
            if np.array_equal(next_snaffle.position, state.opponent_goal.center):
                return True
            snaffle = next_snaffle
            wizard = next_wizard
            future_wizard_target = apply_force(future_wizard_target, thrust=0, destination=future_wizard_target.position,
                                               friction=FRICTION_SNAFFLE, mass=MASS_SNAFFLE)
        return False

    def _capture_snaffle(self, state: GameState, wizard: Wizard, assignments: Dict[EntityId, Snaffle]) -> Union[Move, Spell]:
        snaffle = assignments.get(wizard.id, None)
        if snaffle is not None:
            if self._can_accio(state, wizard, snaffle):
                self.on_accio[snaffle.id] = DURATION_ACCIO + 1
                return Spell.throw_accio(state, snaffle.id)
        else:
            snaffle = self._get_closest_snaffle_to_recapture(state, wizard)
            if snaffle is None:
                snaffle = state.snaffles[0]
        return Move.move_toward(direction=snaffle.position + snaffle.speed)

    def _can_accio(self, state: GameState, wizard: Wizard, snaffle: Snaffle) -> bool:
        possible = state.player_status.magic >= MANA_COST_ACCIO
        possible &= snaffle.id not in self.on_accio
        possible &= distance2(snaffle.position, state.opponent_goal.center) > distance2(wizard.position, state.opponent_goal.center)
        possible &= distance2(wizard.position, snaffle.position) >= 3000 ** 2 or distance2(snaffle.position, state.player_goal.center) < 3000 ** 2 # TODO - should be ball going to goal...
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

    def _shoot_ball(self, state: GameState, wizard: Wizard, free_snaffles: List[Snaffle]):
        field_part = get_field_part(state, wizard)
        if field_part == FieldPart.ATTACK:
            return self._shoot_toward_goal(state, wizard)
        else:
            return self._advance_ball(state, wizard, free_snaffles)

    def _shoot_toward_goal(self, state: GameState, wizard: Wizard):
        # Select the point the closest of the goal to shoot for
        # TODO - avoid the bludgers and the opponents
        goal = state.opponent_goal
        target = min((goal.center, goal.top, goal.bottom), key=lambda p: distance2(p, wizard.position))
        return Move.throw_ball(direction=target - wizard.speed)

    def _advance_ball(self, state: GameState, wizard: Wizard, free_snaffles: List[Snaffle]):

        '''
        # Shoot another ball to make it advance if you can
        threshold = 1000
        goal = state.opponent_goal
        for snaffle in free_snaffles:
            if distance2(snaffle.position, goal.center) < distance2(wizard.position, goal.center):
                if distance2(snaffle.position, wizard.position) < threshold ** 2:
                    if abs(snaffle.position[1] - wizard.position[1]) < threshold / 3:
                        return Move(is_throw=True, direction=snaffle.position - wizard.speed, power=MAX_THROW_POWER)
        '''

        def eval_function(pos: Vector, next_pos: Vector):
            # TODO - just check if something in between - of if it come closer to an opponent wizard?
            # TODO - check the bludgers
            shortest_dist = float('inf')
            for w in state.opponent_wizards:
                shortest_dist = min(shortest_dist, distance2(w.position, next_pos), distance2(w.position + w.speed, next_pos))
            return shortest_dist - distance2(next_pos, state.opponent_goal.center)

        max_metric = float('-inf')
        best_direction = None
        dx = state.opponent_goal.x - wizard.position[0]
        directions = [vector(dx, dx), vector(dx, dx // 2), vector(dx, 0), vector(dx, -dx), vector(dx, -dx // 2)]
        for direction in directions:
            direction = direction + wizard.position - wizard.speed
            snaffle = Snaffle(id=0, position=wizard.position, speed=wizard.speed)
            next_snaffle = apply_force(snaffle, thrust=MAX_THROW_POWER, destination=direction, friction=FRICTION_SNAFFLE, mass=MASS_SNAFFLE)
            metric = eval_function(snaffle.position, next_snaffle.position)
            debug("direction:", direction, "=>", metric)
            if metric > max_metric:
                max_metric = metric
                best_direction = direction
        return Move.throw_ball(direction=best_direction)

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


"""
------------------------------------------------------------------------------------------------------------------------
EXPLORATION BASED AI
------------------------------------------------------------------------------------------------------------------------
"""


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

    while True:
        chrono = Chronometer()
        chrono.start()

        player_status = read_status()
        opponent_status = read_status()
        game_state = read_state(player_status, opponent_status, player_goal, opponent_goal)

        debug("player status:", player_status)
        debug("opponent status:", opponent_status)
        debug("game state:", game_state)

        agent.on_turn_start(game_state)
        for action in agent.get_actions(game_state):
            print(action)

        debug("Time spent:", chrono.spent())


if __name__ == '__main__':
    game_loop(agent=GrabClosestAndShootTowardGoal())
