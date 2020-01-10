from ml.codingame.coders_strike_back_4_genetic import *


track = Track(checkpoints=[np.array([10336,  3367]), np.array([11173,  5410]), np.array([7281, 6642]), np.array([5443, 2814])],
              total_laps=3)


player = [Vehicle(position=np.array([10799.,  3177.]),
                  speed=np.array([0., 0.]),
                  direction=1.4048489275053915,
                  next_checkpoint_id=1,
                  current_lap=0,
                  boost_available=False),
          Vehicle(position=np.array([9873., 3557.]),
                  speed=np.array([0., 0.]),
                  direction=0.959020779005364,
                  next_checkpoint_id=1,
                  current_lap=0,
                  boost_available=False)]


opponent = [Vehicle(position=np.array([11724.,  2798.]),
                    speed=np.array([0., 0.]),
                    direction=1.7786977083244606,
                    next_checkpoint_id=1,
                    current_lap=0,
                    boost_available=False),
            Vehicle(position=np.array([8948., 3936.]),
                    speed=np.array([0., 0.]),
                    direction=0.5850929162378574,
                    next_checkpoint_id=1,
                    current_lap=0,
                    boost_available=False)]


next_player = [Vehicle(position=np.array([10832.,  3374.]), speed=np.array([ 28., 167.]), direction=1.413716694115407, next_checkpoint_id=1, current_lap=0, boost_available=False),
               Vehicle(position=np.array([9988., 3721.]), speed=np.array([ 97., 139.]), direction=0.9599310885968813, next_checkpoint_id=1, current_lap=0, boost_available=False)]


next_opponent = [Vehicle(position=np.array([11707.,  2876.]), speed=np.array([-14.,  66.]), direction=1.780235837034216, next_checkpoint_id=1, current_lap=0, boost_available=False),
                 Vehicle(position=np.array([9015., 3980.]), speed=np.array([56., 37.]), direction=0.593411945678072, next_checkpoint_id=1, current_lap=0, boost_available=False)]



entities = Entities(
    positions=np.stack([v.position for v in itertools.chain(player, opponent)]),
    speeds=np.stack([v.speed for v in itertools.chain(player, opponent)]),
    directions=np.array([v.direction for v in itertools.chain(player, opponent)]),
    radius=np.array([FORCE_FIELD_RADIUS] * 4),
    masses=np.array([1.0] * 4),
    next_checkpoint_id=np.array([v.next_checkpoint_id for v in itertools.chain(player, opponent)]),
    current_lap=np.array([v.current_lap for v in itertools.chain(player, opponent)])
)


simulated = entities.clone()
apply_actions(simulated, actions=[(200, 0), (200, 0)])
simulate_round(simulated, dt=1.0)
print(simulated)

simulated = entities.clone()
simulate_turns(track, simulated, [[(200, 0), (200, 0)]])
print(simulated)

