import numpy as np
import pandas as pd
import simpy


"""
------------------------------------------------------------------------------------------------------------------------
- TAXI SIMULATION: similar to workers taking work from a work-stealing queue
------------------------------------------------------------------------------------------------------------------------
"""


class TaxiSimulationOutcome:
    def __init__(self):
        self.arrived = 0
        self.traveling = 0
        self.serviced = 0
        self.end_of_simulation = 0
        self.end_taxi = []

    @property
    def remaining(self):
        return self.arrived - self.traveling - self.serviced

    def on_passenger_arrival(self, time):
        self.arrived += 1

    def on_passenger_taken(self, time):
        self.traveling += 1

    def on_passenger_dropped(self, time):
        self.traveling -= 1
        self.serviced += 1

    def on_taxi_end(self, time):
        self.end_taxi.append(time)

    def on_simulation_end(self, time):
        self.end_of_simulation = time

    def to_dict(self):
        return {
            'arrived': self.arrived,
            'traveling': self.traveling,
            'serviced': self.serviced,
            'remaining': self.remaining,
            'taxi_end': np.mean(self.end_taxi),
            'simulation_end': self.on_simulation_end
        }


class TaxiSimulation:
    def __init__(self, taxi_count, taxi_end_shift, end_passenger_arrival, passenger_inter_arrival, trip_duration):
        self.taxi_end_shift = taxi_end_shift
        self.end_passenger_arrival = end_passenger_arrival
        self.taxi_count = taxi_count
        self.passenger_inter_arrival = passenger_inter_arrival
        self.trip_duration = trip_duration

    def run(self, until=None):
        # TODO - add a SimulationState class ?
        env = simpy.Environment()
        waiting_passengers = simpy.Container(env, init=0)
        outcome = TaxiSimulationOutcome()

        def passenger_arrival():
            while env.now < self.end_passenger_arrival:
                yield env.timeout(np.random.exponential(scale=self.passenger_inter_arrival))
                waiting_passengers.put(1)
                outcome.on_passenger_arrival(env.now)

        def taxi_cab_routine():
            end_of_day = 0
            while env.now < self.taxi_end_shift:
                yield waiting_passengers.get(1)
                if env.now >= self.taxi_end_shift:      # TODO - use a timeout + disjunction of event to do this...
                    waiting_passengers.put(1)
                    break
                outcome.on_passenger_taken(env.now)
                yield env.timeout(np.random.exponential(scale=self.trip_duration))
                outcome.on_passenger_dropped(env.now)
                end_of_day = env.now
            outcome.on_taxi_end(end_of_day)

        env.process(passenger_arrival())
        for _ in range(self.taxi_count):
            env.process(taxi_cab_routine())
        env.run(until=until)
        outcome.on_simulation_end(env.now)
        return outcome


def taxi_test():
    simulation = TaxiSimulation(taxi_end_shift=10*60, taxi_count=6,
                                end_passenger_arrival=10*60, passenger_inter_arrival=5,
                                trip_duration=30)
    results = [simulation.run(until=None) for _ in range(100)]

    # Using numpy
    print("Average people arrived:", np.mean([result.arrived for result in results]))
    print("Average people served:", np.mean([result.serviced for result in results]))
    print("Average people not served:", np.mean([result.remaining for result in results]))

    # Using pandas
    data_frame = pd.DataFrame(result.to_dict() for result in results)
    # data_frame['remaining'] = data_frame['arrived'] - data_frame['serviced'] - data_frame['traveling']
    print(data_frame.describe())


"""
------------------------------------------------------------------------------------------------------------------------
- BUS SIMULATION: similar to a fixed schedule worker taking everything at once from a fixed size buffer
------------------------------------------------------------------------------------------------------------------------
"""


class BusSimulationOutcome:
    def __init__(self):
        self.arrived = 0
        self.serviced = 0

    def on_passenger_arrival(self, time):
        self.arrived += 1

    def on_passenger_taken(self, time, passenger_count):
        self.serviced += passenger_count

    # TODO - number of people by bus passage
    # TODO - waiting time of passengers (number of bus to wait for?)
    # TODO - number of people that go away (there is such a thing...)

    def to_dict(self):
        return {
            'arrived': self.arrived,
            'serviced': self.serviced
        }


class BusSimulation:
    def __init__(self, bus_capacity, bus_end_shift, bus_inter_arrival, passenger_inter_arrival, end_passenger_arrival):
        self.bus_capacity = bus_capacity
        self.bus_end_shift = bus_end_shift
        self.bus_inter_arrival = bus_inter_arrival
        self.end_passenger_arrival = end_passenger_arrival
        self.passenger_inter_arrival = passenger_inter_arrival

    def run(self, until=None):
        env = simpy.Environment()
        waiting_passengers = simpy.Container(env, init=0)
        outcome = BusSimulationOutcome()

        def passenger_arrival():
            while env.now < self.end_passenger_arrival:
                yield env.timeout(np.random.exponential(scale=self.passenger_inter_arrival))
                outcome.on_passenger_arrival(env.now)
                waiting_passengers.put(1)

        def bus_routine():
            while env.now < self.bus_end_shift:
                passenger_count = min(waiting_passengers.level, self.bus_capacity)
                if passenger_count > 0:
                    waiting_passengers.get(passenger_count)
                outcome.on_passenger_taken(env.now, passenger_count)
                yield env.timeout(np.random.exponential(scale=self.bus_inter_arrival))

        env.process(passenger_arrival())
        env.process(bus_routine())
        env.run(until=until)
        return outcome


def bus_test():
    simulation = BusSimulation(bus_capacity=40, bus_inter_arrival=30, bus_end_shift=10*60,
                               passenger_inter_arrival=5, end_passenger_arrival=10*60)
    results = [simulation.run(until=None) for _ in range(100)]
    data_frame = pd.DataFrame(result.to_dict() for result in results)
    print(data_frame.describe())


"""
------------------------------------------------------------------------------------------------------------------------
Running the tests 
------------------------------------------------------------------------------------------------------------------------
"""


taxi_test()
print()
bus_test()
