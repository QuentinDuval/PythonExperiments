import collections
import dataclasses
import functools
import itertools
import numpy as np
import pandas as pd
import simpy


@dataclasses.dataclass
class TaxiSimulationOutcome:
    arrived: int = 0
    traveling: int = 0
    serviced: int = 0

    @property
    def remaining(self):
        return self.arrived - self.traveling - self.serviced

    def on_passenger_arrival(self):
        self.arrived += 1

    def on_passenger_taken(self):
        self.traveling += 1

    def on_passenger_dropped(self):
        self.traveling -= 1
        self.serviced += 1

    def to_dict(self):
        return dict(dataclasses.asdict(self), remaining=self.remaining)


class TaxiSimulation:
    def __init__(self, end_shift, taxi_count, passenger_inter_arrival, trip_duration):
        self.end_shift = end_shift
        self.taxi_count = taxi_count
        self.passenger_inter_arrival = passenger_inter_arrival
        self.trip_duration = trip_duration

    def run(self):
        env = simpy.Environment()
        waiting_passengers = simpy.Container(env, init=0)
        outcome = TaxiSimulationOutcome()

        def passenger_arrival():
            for _ in itertools.count():
                delay = np.random.exponential(scale=self.passenger_inter_arrival)
                yield env.timeout(delay)
                waiting_passengers.put(1)
                outcome.on_passenger_arrival()

        def taxi_cab_routine():
            while True:
                yield waiting_passengers.get(1)
                if env.now >= self.end_shift:
                    waiting_passengers.put(1)
                    break
                outcome.on_passenger_taken()
                yield env.timeout(np.random.exponential(scale=self.trip_duration))
                outcome.on_passenger_dropped()

        env.process(passenger_arrival())
        for _ in range(self.taxi_count):
            env.process(taxi_cab_routine())
        env.timeout(self.end_shift)
        env.run(until=self.end_shift)
        return outcome


def taxi_test():
    simulation = TaxiSimulation(end_shift=10*60, taxi_count=6, passenger_inter_arrival=5, trip_duration=30)
    results = [simulation.run() for _ in range(100)]

    # Using numpy
    print("Average people arrived:", np.mean([result.arrived for result in results]))
    print("Average people served:", np.mean([result.serviced for result in results]))
    print("Average people in flight (taxi overtime):", np.mean([result.traveling for result in results]))
    print("Average people not served:", np.mean([result.remaining for result in results]))

    # Using pandas
    data_frame = pd.DataFrame(result.to_dict() for result in results)
    # data_frame['remaining'] = data_frame['arrived'] - data_frame['serviced'] - data_frame['traveling']
    print(data_frame.describe())


taxi_test()
