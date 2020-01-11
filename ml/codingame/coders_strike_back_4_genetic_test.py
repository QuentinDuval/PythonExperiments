from ml.codingame.coders_strike_back_4_genetic import *


track = Track(
    checkpoints=[np.array([13595,  7615]), np.array([12460,  1373]), np.array([10555,  6003]), np.array([3601, 5155])],
    total_laps=3
)


def scenario_0():
    # TURN 105->106
    # NO COLLISIONS (still cannot reproduce what happens in the game...)

    entities = Entities.empty(size=1)
    entities.positions = np.array([[2776., 5235.]])
    entities.speeds = np.array([[-52., -519.]])
    entities.directions = np.array([[5.55014702]])

    """
    PLAYER ENTITIES
    [[ 2776.  5235.]
     [13436.  4046.]]
    [[ -52. -519.]
     [-521.  591.]]
    action: 200.0 0.3141592653589793
    action: 200.0 0.0
    """

    simulated = entities.clone()
    apply_actions(simulated, actions=[(200.0, 0.3141592653589793)])
    simulate_round(simulated, dt=1.0)
    print(simulated)

    simulated = entities.clone()
    simulate_turns(track, simulated, actions_by_turn=[[(200.0, 0.3141592653589793)]])
    print(simulated)

    """
    PLAYER ENTITIES
    [[ 2907.  4635.]
     [12747.  4746.]]
    [[ 111. -510.]
     [-585.  594.]]
    action: 200.0 0.3141592653589793
    action: 200.0 0.3141592653589793
    """


def scenario_1():
    # TURN 127

    entities = Entities.empty(size=2)
    entities.positions = np.array([[3217., 4104.], [4056., 1865.]])
    entities.speeds = np.array([[263., -451.], [ 358.,  297.]])
    entities.directions = np.array([6.17846555, 1.6231562 ])

    """
    PLAYER ENTITIES
    [[3217. 4104.]
     [4056. 1865.]]
    [[ 263. -451.]
     [ 358.  297.]]
    action: 200.0 0.3141592653589793
    action: 200.0 0.3141592653589793
    """

    apply_actions(entities, actions=[(200.0, 0.3141592653589793), (200.0, 0.3141592653589793)])
    simulate_round(entities, dt=1.0)
    print(entities)

    """
    PLAYER ENTITIES
    [[3676. 3695.]
     [4342. 2349.]]
    [[ 389. -348.]
     [ 243.  411.]]
    action: 200.0 0.3141592653589793
    action: 200.0 0.0
    """


def scenario_2():
    # TURN 128

    entities = Entities.empty(size=2)
    entities.positions = np.array([[3676., 3695.], [4342., 2349.]])
    entities.speeds = np.array([[389., -348.], [243., 411.]])
    entities.directions = np.array([0.20943951, 1.93731547])

    """
    PLAYER ENTITIES
    [[3676. 3695.]
     [4342. 2349.]]
    [[ 389. -348.]
     [ 243.  411.]]
    action: 200.0 0.3141592653589793
    action: 200.0 0.0
    """

    apply_actions(entities, actions=[(200.0, 0.3141592653589793), (200.0, 0.0)])
    simulate_round(entities, dt=1.0)
    print(entities)

    # TODO - no detection of collisions here

    """
    PLAYER ENTITIES
    [[4132. 3650.]
     [4620. 2744.]]
    [[ 110.  489.]
     [ 512. -192.]]
    BAD PREDICTION
    --------------------
    PRED positions: [[4238. 3447.]
     [4513. 2947.]]
    GOT positions: [[4132. 3650.]
     [4620. 2744.]]
    PRED speeds: [[ 477. -210.]
     [ 145.  508.]]
    GOT speeds: [[ 110.  489.]
     [ 512. -192.]]
    PRED angles: [0.52359878 1.93731547]
    GOT angles: [0.52359878 1.93731547]
    action: 200.0 0.3141592653589793
    action: 200.0 0.0
    """


# TODO - encode this as unit tests

# scenario_0()
# print()
# print("-" * 50)
# print()
scenario_1()
print()
print("-" * 50)
print()
scenario_2()
