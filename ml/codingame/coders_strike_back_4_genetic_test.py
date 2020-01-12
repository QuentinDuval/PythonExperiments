from ml.codingame.coders_strike_back_4_genetic import *


def test_scenario_0():
    """
    Simulation of a single unit (no collisions)
    """

    entities = Entities.empty(size=1)
    entities.positions = np.array([[2776., 5235.]])
    entities.speeds = np.array([[-52., -519.]])
    entities.directions = np.array([[5.55014702]])

    apply_actions(entities, thrusts=np.array([200.]), diff_angles=np.array([0.3141592653589793]))
    simulate_movements(entities, dt=1.0)
    assert np.array_equal(entities.positions, np.array([[2907., 4635.]]))
    assert np.array_equal(entities.speeds, np.array([[111., -510.]]))


def test_scenario_1():
    """
    Simulation of two units (no collisions)
    """

    entities = Entities.empty(size=2)
    entities.positions = np.array([[3217., 4104.], [4056., 1865.]])
    entities.speeds = np.array([[263., -451.], [ 358.,  297.]])
    entities.directions = np.array([6.17846555, 1.6231562 ])

    apply_actions(entities, thrusts=np.array([200., 200.]), diff_angles=np.array([0.3141592653589793, 0.3141592653589793]))
    simulate_movements(entities, dt=1.0)
    assert np.array_equal(entities.positions, np.array([[3676., 3695.], [4342., 2349.]]))
    assert np.array_equal(entities.speeds, np.array([[389., -348.], [243., 411.]]))


def test_scenario_2():
    """
    Simulation of two units (with collision)
    """

    entities = Entities.empty(size=2)
    entities.positions = np.array([[3676., 3695.], [4342., 2349.]])
    entities.speeds = np.array([[389., -348.], [243., 411.]])
    entities.directions = np.array([0.20943951, 1.93731547])

    apply_actions(entities, thrusts=np.array([200., 200.]), diff_angles=np.array([0.3141592653589793, 0.]))
    simulate_movements(entities, dt=1.0)
    assert np.array_equal(entities.positions, np.array([[4132., 3650.], [4620., 2744.]]))
    assert np.array_equal(entities.speeds, np.array([[110., 489.], [512., -192.]]))


def test_simulation_performance():
    nb_scenario = 1000
    nb_action = 4

    np.random.seed(1)
    thrusts = np.random.uniform(0., 200., size=(nb_scenario, nb_action, 2))
    angles = np.random.choice([-MAX_TURN_RAD, 0, MAX_TURN_RAD], replace=True, size=(nb_scenario, nb_action, 2))
    track = Track(
        checkpoints=[np.array([13595, 7615]), np.array([12460, 1373]), np.array([10555, 6003]), np.array([3601, 5155])],
        total_laps=3)

    entities = Entities.empty(size=4)
    entities.positions = np.array([[3676., 3695.], [4342., 2349.], [3217., 4104.], [4056., 1865.]])
    entities.speeds = np.array([[389., -348.], [243., 411.], [263., -451.], [ 358.,  297.]])
    entities.directions = np.array([0.20943951, 1.93731547, 6.17846555, 1.6231562 ])

    chrono = Chronometer()
    chrono.start()
    for i in range(thrusts.shape[0]):
        simulated = entities.clone()
        simulate_turns(track, simulated, thrusts[i], angles[i])
    print("time spent:", chrono.spent())


test_scenario_0()
test_scenario_1()
test_scenario_2()
test_simulation_performance()
