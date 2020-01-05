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
    s1 = Snaffle(id=1, position=vector(RIGHT_GOAL.x - 100, RIGHT_GOAL.y_lo - 10), speed=(0, 0))
    s2 = Snaffle(id=1, position=vector(RIGHT_GOAL.x + 100, RIGHT_GOAL.y_hi + 10), speed=(0, 0))
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
    s1 = Snaffle(id=1, position=vector(LEFT_GOAL.x + 100, LEFT_GOAL.y_lo - 10), speed=(0, 0))
    s2 = Snaffle(id=1, position=vector(LEFT_GOAL.x - 100, LEFT_GOAL.y_hi + 10), speed=(0, 0))
    assert not intersect_goal(s1, s2, goal=RIGHT_GOAL)
    assert intersect_goal(s1, s2, goal=LEFT_GOAL)


def run_tests():
    test_intersect_goal()


run_tests()
