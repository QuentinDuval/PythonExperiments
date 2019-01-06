import collections
import functools
import itertools
import numpy as np
import pandas as pd
import simpy


TaxiSimulationOutcome = collections.namedtuple(
    "TaxiSimulationOutcome", ["serviced", "traveling", "remaining"]
)


class TaxiSimulation:
    def __init__(self, end_shift, taxi_count, passenger_inter_arrival, trip_duration):
        self.end_shift = end_shift
        self.taxi_count = taxi_count
        self.passenger_inter_arrival = passenger_inter_arrival
        self.trip_duration = trip_duration

    class Resources:
        def __init__(self, env):
            self.waiting_passengers = simpy.Container(env, init=0)
            self.traveling_passengers = 0
            self.serviced_passengers = 0

    def run(self):
        env = simpy.Environment()
        resources = self.Resources(env)
        env.process(self.passenger_arrival(env, resources))
        for _ in range(self.taxi_count):
            env.process(self.taxi_cab_routine(env, resources))
        env.timeout(self.end_shift)
        env.run(until=self.end_shift)
        return TaxiSimulationOutcome(serviced=resources.serviced_passengers,
                                     traveling=resources.traveling_passengers,
                                     remaining=resources.waiting_passengers.level)

    def passenger_arrival(self, env, resources):
        for i in itertools.count():
            delay = np.random.exponential(scale=self.passenger_inter_arrival)
            yield env.timeout(delay)
            # print("Passenger", i, "at", env.now, "after delay", delay)
            resources.waiting_passengers.put(1)

    def taxi_cab_routine(self, env, resources):
        while True:
            yield resources.waiting_passengers.get(1)
            if env.now >= self.end_shift:
                resources.waiting_passengers.put(1)
                break
            # print("Taking passenger at", env.now)
            resources.traveling_passengers += 1
            yield env.timeout(np.random.exponential(scale=self.trip_duration))
            resources.traveling_passengers -= 1
            resources.serviced_passengers += 1
            # print("Dropping passenger at", env.now)
        # print("> Taxi going back home at", last_trip_end)

def taxi_test():
    simulation = TaxiSimulation(end_shift=10*60, taxi_count=6, passenger_inter_arrival=5, trip_duration=30)
    results = [simulation.run() for _ in range(100)]

    # Using numpy
    print("Average people served:", np.mean([result.serviced for result in results]))
    print("Average people in flight (taxi overtime):", np.mean([result.traveling for result in results]))
    print("Average people not served:", np.mean([result.remaining for result in results]))

    # Using pandas
    data_frame = pd.DataFrame(results)
    print(data_frame.describe())


taxi_test()
