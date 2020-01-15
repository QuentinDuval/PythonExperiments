from ml.codingame.coders_strike_back_4_genetic import *

import cProfile


track = Track(
        checkpoints=[np.array([13595, 7615]),
                     np.array([12460, 1373]),
                     np.array([10555, 6003]),
                     np.array([3601, 5155])],
        total_laps=3)


def test_scenario_0():
    """
    Simulation of a single unit (no collisions)
    """

    entities = Entities.empty(size=1)
    entities.positions = np.array([[2776., 5235.]])
    entities.speeds = np.array([[-52., -519.]])
    entities.directions = np.array([[5.55014702]])

    apply_actions(entities, thrusts=np.array([200.]), diff_angles=np.array([0.3141592653589793]))
    simulate_movements(track, entities, dt=1.0)
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

    apply_actions(entities,
                  thrusts=np.array([200., 200.]),
                  diff_angles=np.array([0.3141592653589793, 0.3141592653589793]))
    simulate_movements(track, entities, dt=1.0)
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
    simulate_movements(track, entities, dt=1.0)
    assert np.array_equal(entities.positions, np.array([[4132., 3650.], [4620., 2744.]]))
    assert np.array


def test_scenario_3():
    track = Track(
        checkpoints=[
            np.array([1000.,  0.]),
            np.array([2000., 0.])],
        total_laps=3)

    # Going toward the checkpoint
    entities = Entities.empty(size=1)
    entities.positions = np.array([[0., 0.]])
    entities.speeds = np.array([[2000., 0.]])
    entities.directions = np.array([0.])
    entities.next_progress_id[0] = 0
    assert 0.2 == find_cp_collision(track, entities, 0, dt=1.0)

    # Going away from the checkpoint
    entities = Entities.empty(size=1)
    entities.positions = np.array([[3000., 0.]])
    entities.speeds = np.array([[2000., 0.]])
    entities.directions = np.array([0.])
    entities.next_progress_id[0] = 0
    assert float('inf') == find_cp_collision(track, entities, 0, dt=1.0)

    # Starting inside the checkpoint
    '''
    entities = Entities.empty(size=1)
    entities.positions = np.array([[600., 0.]])
    entities.speeds = np.array([[2000., 0.]])
    entities.directions = np.array([0.])
    entities.next_progress_id[0] = 0
    assert 0.0 == find_cp_collision(track, entities, 0, dt=1.0)
    '''

    # Going very close outside the checkpoint
    entities = Entities.empty(size=1)
    entities.positions = np.array([[0., 600.]])
    entities.speeds = np.array([[2000., 0.]])
    entities.directions = np.array([0.])
    entities.next_progress_id[0] = 0
    assert float('inf') == find_cp_collision(track, entities, 0, dt=1.0)

    # Going very close inside the checkpoint
    entities = Entities.empty(size=1)
    entities.positions = np.array([[0., 599.]])
    entities.speeds = np.array([[2000., 0.]])
    entities.directions = np.array([0.])
    entities.next_progress_id[0] = 0
    assert float('inf') != find_cp_collision(track, entities, 0, dt=1.0)


def test_full_game():
    """
    Simulation of one unit, trying to find the best path on a track
    """

    track = Track(
        checkpoints=[
            np.array([10214,  4913]),
            np.array([6081, 2227]),
            np.array([3032, 5200]),
            np.array([6273, 7759]),
            np.array([14122,  7741]),
            np.array([13853,  1220])],
        total_laps=3)

    entities = Entities.empty(size=2)
    entities.positions = np.array([[9942., 5332.], [10486.,  4494.]])
    entities.speeds = np.array([[0., 0.], [0., 0.]])
    entities.directions = np.array([3.81888678, 3.61688549])
    entities.next_progress_id = np.array([1, 1])

    agent = GeneticAgent(track)
    total_turns = 0
    while entities.next_progress_id.max() < len(track) * 3 + 1 and total_turns < 400:    # 2 full turns completed
        total_turns += 1
        actions = agent.get_action(entities)
        simulate_turns(track, entities,
                       thrusts=np.array([[actions[i].thrust for i in range(2)]]),
                       diff_angles=np.array([[actions[i].angle for i in range(2)]]))
    print("turns:", total_turns)
    print("progress:", entities.next_progress_id[0], entities.next_progress_id[1])


def test_simulation_performance(profiler: bool):
    nb_scenario = 1000
    nb_action = 4

    np.random.seed(1)
    thrusts = np.random.uniform(0., 200., size=(nb_scenario, nb_action, 4))
    angles = np.random.choice([-MAX_TURN_RAD, 0, MAX_TURN_RAD], replace=True, size=(nb_scenario, nb_action, 4))

    entities = Entities.empty(size=4)
    entities.positions = np.array([[3676., 3695.], [4342., 2349.], [3217., 4104.], [4056., 1865.]])
    entities.speeds = np.array([[389., -348.], [243., 411.], [263., -451.], [ 358.,  297.]])
    entities.directions = np.array([0.20943951, 1.93731547, 6.17846555, 1.6231562 ])

    def test_loop():
        for i in range(thrusts.shape[0]):
            simulated = entities.clone()
            simulate_turns(track, simulated, thrusts[i], angles[i])

    if profiler:
        profiler = cProfile.Profile()
        profiler.runcall(test_loop)
        profiler.print_stats(sort=1)
    else:
        chrono = Chronometer()
        chrono.start()
        test_loop()
        time_spent = chrono.spent()
        print("time spent:", time_spent, "ms")
        print("time by scenario:", time_spent / nb_scenario, "ms")
        print("scenario by turn:", nb_scenario / time_spent * RESPONSE_TIME * 0.8, "(", RESPONSE_TIME * 0.8, "ms)")


test_scenario_0()
test_scenario_1()
test_scenario_2()
test_scenario_3()

# test_full_game()
test_simulation_performance(profiler=True)
