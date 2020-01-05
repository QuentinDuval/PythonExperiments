from ml.codingame.fantastic_bits import *


def test_intersect_goal():
    # Direct straight shot (RIGHT)
    s1 = Snaffle(id=1, position=vector(15009, RIGHT_GOAL.center[1]), speed=vector(0, 0))
    s2 = Snaffle(id=1, position=vector(16009, RIGHT_GOAL.center[1]), speed=vector(0, 0))
    assert intersect_goal(s1, s2, goal=RIGHT_GOAL)
    assert not intersect_goal(s1, s2, goal=LEFT_GOAL)

    # Shot from one post to the other (RIGHT)
    s1 = Snaffle(id=1, position=vector(15009, RIGHT_GOAL.y_lo), speed=vector(0, 0))
    s2 = Snaffle(id=1, position=vector(16009, RIGHT_GOAL.y_hi), speed=vector(0, 0))
    assert intersect_goal(s1, s2, goal=RIGHT_GOAL)
    assert not intersect_goal(s1, s2, goal=LEFT_GOAL)

    # Shot from diagonal
    s1 = Snaffle(id=1, position=vector(RIGHT_GOAL.x - 100, RIGHT_GOAL.y_lo - 10), speed=vector(0, 0))
    s2 = Snaffle(id=1, position=vector(RIGHT_GOAL.x + 100, RIGHT_GOAL.y_hi + 10), speed=vector(0, 0))
    assert intersect_goal(s1, s2, goal=RIGHT_GOAL)
    assert not intersect_goal(s1, s2, goal=LEFT_GOAL)

    # Direct straight shot (LEFT)
    s1 = Snaffle(id=1, position=vector(10, LEFT_GOAL.center[1]), speed=vector(0, 0))
    s2 = Snaffle(id=1, position=vector(-10, LEFT_GOAL.center[1]), speed=vector(0, 0))
    assert not intersect_goal(s1, s2, goal=RIGHT_GOAL)
    assert intersect_goal(s1, s2, goal=LEFT_GOAL)

    # Shot from one post to the other (LEFT)
    s1 = Snaffle(id=1, position=vector(10, LEFT_GOAL.y_lo), speed=vector(0, 0))
    s2 = Snaffle(id=1, position=vector(-10, LEFT_GOAL.y_hi), speed=vector(0, 0))
    assert not intersect_goal(s1, s2, goal=RIGHT_GOAL)
    assert intersect_goal(s1, s2, goal=LEFT_GOAL)

    # Shot from diagonal
    s1 = Snaffle(id=1, position=vector(LEFT_GOAL.x + 100, LEFT_GOAL.y_lo - 10), speed=vector(0, 0))
    s2 = Snaffle(id=1, position=vector(LEFT_GOAL.x - 100, LEFT_GOAL.y_hi + 10), speed=vector(0, 0))
    assert not intersect_goal(s1, s2, goal=RIGHT_GOAL)
    assert intersect_goal(s1, s2, goal=LEFT_GOAL)


def test_apply_force():
    # Test un-impaired move
    s1 = Snaffle(id=1, position=vector(100, 10), speed=vector(0, 20))
    s2 = apply_force(s1, thrust=0, destination=s1.position, friction=FRICTION_SNAFFLE, mass=MASS_SNAFFLE)
    assert s2 == Snaffle(id=1, position=vector(100, 30), speed=vector(0, 15))

    # Test un-impaired move
    s1 = Snaffle(id=1, position=vector(100, 10), speed=vector(10, 10))
    s2 = apply_force(s1, thrust=0, destination=s1.position, friction=FRICTION_SNAFFLE, mass=MASS_SNAFFLE)
    assert s2 == Snaffle(id=1, position=vector(110, 20), speed=vector(7, 7))

    # Test rebound on the wall Y = 0
    s1 = Snaffle(id=1, position=vector(100, 10), speed=vector(0, -20))
    s2 = apply_force(s1, thrust=0, destination=s1.position, friction=FRICTION_SNAFFLE, mass=MASS_SNAFFLE)
    assert s2 == Snaffle(id=1, position=vector(100, 10), speed=vector(0, 15))

    # Test rebound on the wall Y = 0
    s1 = Snaffle(id=1, position=vector(100, 10), speed=vector(10, -20))
    s2 = apply_force(s1, thrust=0, destination=s1.position, friction=FRICTION_SNAFFLE, mass=MASS_SNAFFLE)
    assert s2 == Snaffle(id=1, position=vector(110, 10), speed=vector(7, 15))

    # Test rebound on the wall Y = HEIGHT
    s1 = Snaffle(id=1, position=vector(100, HEIGHT-10), speed=vector(10, 20))
    s2 = apply_force(s1, thrust=0, destination=s1.position, friction=FRICTION_SNAFFLE, mass=MASS_SNAFFLE)
    assert s2 == Snaffle(id=1, position=vector(110, HEIGHT-10), speed=vector(7, -15))


def run_tests():
    test_intersect_goal()
    test_apply_force()


run_tests()
