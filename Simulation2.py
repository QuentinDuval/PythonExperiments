import collections
import functools
import itertools
import numpy as np
import simpy


class TaxiSimulation:
    def __init__(self, end_shift, taxi_count, passenger_inter_arrival, trip_duration):
        self.end_shift = end_shift
        self.taxi_count = taxi_count
        self.passenger_inter_arrival = passenger_inter_arrival
        self.trip_duration = trip_duration

    def run(self):
        env = simpy.Environment()
        passengers = simpy.Container(env, init=0)
        env.process(self.passenger_arrival(env, passengers))
        for _ in range(self.taxi_count):
            env.process(self.taxi_cab_routine(env, passengers))
        env.timeout(self.end_shift)
        env.run(until=self.end_shift)
        # print("=> Remaining amount of passengers", passengers.level)
        # TODO - beware some passengers are "in flight"
        return passengers.level

    def passenger_arrival(self, env, passengers):
        for i in itertools.count():
            delay = np.random.exponential(scale=self.passenger_inter_arrival)
            yield env.timeout(delay)
            # print("Passenger", i, "at", env.now, "after delay", delay)
            passengers.put(1)

    def taxi_cab_routine(self, env, passengers):
        last_trip_end = 0
        while True:
            yield passengers.get(1)
            if env.now >= self.end_shift:
                passengers.put(1)
                break
            # print("Taking passenger at", env.now)
            yield env.timeout(np.random.exponential(scale=self.trip_duration))
            # print("Dropping passenger at", env.now)
            last_trip_end = env.now
        # print("> Taxi going back home at", last_trip_end)


def taxi_test():
    simulation = TaxiSimulation(end_shift=10*60, taxi_count=6, passenger_inter_arrival=5, trip_duration=30)
    times = [simulation.run() for _ in range(100)]
    print("Average people not served:", np.mean(times)) # TODO - give percentage instead, use pandas data frame (and output more data)
    print("Standard deviation:", np.std(times))


taxi_test()
