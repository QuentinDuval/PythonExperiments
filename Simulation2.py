import collections
import functools
import itertools
import numpy as np
import simpy


Passenger = collections.namedtuple(
    'Passenger', ['identifier']
)


class TaxiSimulation:
    def __init__(self):
        self.env = simpy.Environment()
        self.passengers = simpy.Container(self.env, init=0)

    def run(self, end_shift, taxi_count):
        self.env.process(self.passenger_arrival())
        for _ in range(taxi_count):
            self.env.process(self.taxi_cab_routine(end_shift))
        self.env.run(until=end_shift)
        print("Remaining amoung of passengers", self.passengers.level)

    def passenger_arrival(self):
        for i in itertools.count():
            delay = np.random.exponential(scale=5)
            yield self.env.timeout(delay)
            print("Passenger", i, "at", self.env.now, "after delay", delay)
            self.passengers.put(1)

    def taxi_cab_routine(self, end_shift):
        while self.env.peek() < end_shift:
            yield self.passengers.get(1)
            print("Taking passenger at", self.env.now)
            yield self.env.timeout(np.random.exponential(scale=30))
            print("Dropping passenger at", self.env.now)


def taxi_test():
    simulation = TaxiSimulation()
    simulation.run(end_shift=10 * 60, taxi_count=6)


taxi_test()
