from ml.codingame.fantastic_bits_ga import *


def test_collision_detection():
    entities = Entities(
        positions=np.array([[0., 0.], [3., 1.]]),
        speeds=np.array([[6., 0.], [0., 0.]]),
        radius=np.array([1., 1.]),
        masses=np.array([1., 1.]),
        frictions=np.array([0.75, 0.75])
    )
    assert (3 - math.sqrt(3)) / 6 == find_collision(entities, 0, 1, dt=1.0)

    simulate_collisions(entities, dt=1.0)
    print(entities)


def run_tests():
    test_collision_detection()


if __name__ == '__main__':
    run_tests()

