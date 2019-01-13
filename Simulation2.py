from collections import *
import math
import matplotlib.pyplot as pyplot
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
        waiting_passengers = simpy.Container(env, init=0) # TODO use simpy.Store to store latency
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
- PERFORMANCE SIMULATION:
  * New items to handle come at a given rate
  * Treat all items (up to 1000 to limit memory) at once
  * Send one notification / save DB for each item handled (simulate round trip!)

See the impact of batching of save / notification
See the impact of multi-threading of save / notification
------------------------------------------------------------------------------------------------------------------------
"""


class PerformanceLog:
    def __init__(self):
        self.time = []
        self.timing = []
        self.item_count = []
        self.mean_latency = []
        self.max_latency = []

    def to_dict(self):
        return {
            'time': self.time,
            'timing': self.timing,
            'mean_latency': self.mean_latency,
            'max_latency': self.max_latency,
            'item_count': self.item_count
        }


class PerformanceTest:
    def __init__(self, input_inter_arrival):
        self.input_inter_arrival = input_inter_arrival
        self.round_trip_duration = 1
        self.poll_request_duration = 5
        self.make_handled_duration = 5
        self.polling_delay = 100
        self.max_chunk = 1000

    def run(self, until=None):
        env = simpy.Environment()
        item_queue = deque()
        simulation_log = PerformanceLog()

        def item_arrival():
            while True:
                yield env.timeout(np.random.exponential(scale=self.input_inter_arrival))
                item_queue.append(env.now)

        def read_db():
            delay = self.poll_request_duration * (1 + math.log(1 + len(item_queue)) / 10)
            yield env.timeout(np.random.exponential(scale=delay))
            item_count = min(len(item_queue), self.max_chunk)
            return [item_queue.popleft() for _ in range(item_count)]

        def send_update():
            yield env.timeout(np.random.exponential(scale=self.round_trip_duration))

        def update_entries(items):
            delay = self.make_handled_duration * (1 + math.log(1 + len(items)) / 10)
            yield env.timeout(np.random.exponential(scale=delay))

        def consumer():
            while True:
                start = env.now

                # TODO - To parallelize, use AllOf? (https://simpy.readthedocs.io/en/latest/topical_guides/events.html)
                # TODO - Or use another process (thread pool) consuming the messages?
                # TODO - Simulate the overload of the destination server

                latencies = []
                items = yield from read_db()
                if items:
                    for item in items:
                        yield from send_update()
                        latencies.append(env.now - item)
                    yield from update_entries(items)

                simulation_log.time.append(env.now)
                simulation_log.item_count.append(len(items))
                simulation_log.timing.append(env.now - start)
                simulation_log.mean_latency.append(np.mean(latencies) if latencies else 0)
                simulation_log.max_latency.append(np.max(latencies) if latencies else 0)

                """
                if env.now - start < self.polling_delay:
                    yield env.timeout(self.polling_delay - env.now + start)
                """
                yield env.timeout(self.polling_delay)

        env.process(item_arrival())
        env.process(consumer())
        env.run(until=until)
        return simulation_log


def performance_test():
    input_inter_arrivals = np.array([20, 10, 5, 2, 1.5, 1.25, 1.15])

    results = []
    for input_inter_arrival in input_inter_arrivals:
        simulation = PerformanceTest(input_inter_arrival=input_inter_arrival)
        result = simulation.run(until=2 * 60 * 1000)
        data_frame = pd.DataFrame(result.to_dict())
        # print(data_frame.describe())
        results.append(data_frame)

    pyplot.subplot(4, 1, 1)
    for result in results:
        pyplot.plot(result['time'], result['item_count'])
    pyplot.ylabel('Item count')

    pyplot.subplot(4, 1, 2)
    for result in results:
        pyplot.plot(result['time'], result['timing'])
    pyplot.ylabel('Timing')

    pyplot.subplot(4, 1, 3)
    for result in results:
        pyplot.plot(result['time'], result['mean_latency'])
    pyplot.ylabel('Latency')

    pyplot.subplot(4, 1, 4)
    pyplot.plot(1000 / input_inter_arrivals, [result['mean_latency'].mean() for result in results])
    pyplot.plot(1000 / input_inter_arrivals, [result['max_latency'].max() for result in results])
    pyplot.xlabel('Latency / Arrival rate')

    pyplot.show()


"""
------------------------------------------------------------------------------------------------------------------------
Running the tests 
------------------------------------------------------------------------------------------------------------------------
"""


# taxi_test()
print()
# bus_test()
print()
performance_test()
