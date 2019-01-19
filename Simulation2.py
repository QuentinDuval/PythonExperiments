from collections import *
import datetime
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

        self.message_emissions = []
        self.message_receptions = []
        self.message_latencies = []

    def to_chunk_report(self):
        df = pd.DataFrame({
            'time': self.time,
            'timing': self.timing,
            'mean_latency': self.mean_latency,
            'max_latency': self.max_latency,
            'item_count': self.item_count
        })
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        return df

    def to_message_report(self):
        df = pd.DataFrame({
            'emission': self.message_emissions,
            'reception': self.message_receptions,
            'latency': self.message_latencies
        })
        df['emission'] = pd.to_datetime(df['emission'], unit='ms')
        df['reception'] = pd.to_datetime(df['reception'], unit='ms')
        return df


class UniformItemArrival:
    def __init__(self, input_inter_arrival):
        self.input_inter_arrival = input_inter_arrival
        self.total_duration = 10 * 60 * 1000

    def __call__(self, env, db_lock, item_queue):
        pending = []
        while True:
            yield env.timeout(np.random.exponential(scale=self.input_inter_arrival))
            pending.append(env.now)
            if db_lock.count == 0:
                item_queue.extend(pending)
                pending.clear()


class BurstyItemArrival:
    def __init__(self, input_inter_arrival, burst):
        self.input_inter_arrival = input_inter_arrival
        self.burst = burst
        self.total_duration = 10 * 60 * 1000

    def __call__(self, env, db_lock, item_queue):
        pending = []
        while True:
            yield env.timeout(np.random.exponential(scale=self.input_inter_arrival * self.burst))
            pending.extend(env.now for _ in range(self.burst))
            if db_lock.count == 0:
                item_queue.extend(pending)
                pending.clear()


class RealItemArrival:
    def __init__(self, real_values):
        self.inter_arrivals = map(lambda x, y: y - x, real_values, real_values[1:])
        self.total_duration = (real_values.iloc[-1] - real_values.iloc[0]) / np.timedelta64(1, 'ms')
        print(self.total_duration)

    def __call__(self, env, db_lock, item_queue):
        for diff in self.inter_arrivals:
            millis = diff / np.timedelta64(1, 'ms')
            if millis > 0:
                yield env.timeout(millis)
            item_queue.append(env.now)


class PerformanceTest:
    def __init__(self, arrival_strategy):
        self.arrival_strategy = arrival_strategy
        self.ack_round_trip_duration = 2
        self.ack_window_size = 1
        self.parallelization = 1
        self.select_request_duration = 5    # TODO - measure by doing request on DB
        self.update_request_duration = 10   # TODO - measure by doing request on DB
        self.polling_delay = 100
        self.max_chunk = 1000               # If you try 40, you will get interesting results

    def run(self, until=None):
        env = simpy.Environment()
        db_lock = simpy.Resource(env, capacity=1)
        item_queue = deque()
        simulation_log = PerformanceLog()

        def read_db():
            delay = self.select_request_duration * (1 + len(item_queue) / 100)
            yield env.timeout(np.random.exponential(scale=delay))
            item_count = min(len(item_queue), self.max_chunk)
            return [item_queue.popleft() for _ in range(item_count)]

        def send_update():
            round_trips = [np.random.exponential(scale=self.ack_round_trip_duration)
                           for _ in range(self.parallelization)]
            yield env.timeout(np.max(round_trips))

        def update_entries(items):
            with db_lock.request():
                delay = self.update_request_duration * (1 + math.log(1 + len(items)) / 10)
                yield env.timeout(np.random.exponential(scale=delay))

        def wait_for_polling(start_time):
            """
            if env.now - start_time < self.polling_delay:
                yield env.timeout(self.polling_delay - (env.now - start_time))
            """
            yield env.timeout(self.polling_delay)

        def consumer():
            while True:
                start_time = env.now

                # TODO - To parallelize, use AllOf? (https://simpy.readthedocs.io/en/latest/topical_guides/events.html)
                # TODO - Or use another process (thread pool) consuming the messages?
                # TODO - Simulate the overload of the destination server

                emissions = []
                latencies = []
                receptions = []
                items = yield from read_db()
                if items:
                    group_size = self.ack_window_size * self.parallelization
                    for start in range(0, len(items), group_size):
                        yield from send_update()
                        end = min(len(items), start+group_size)
                        for item in items[start:end]:
                            emissions.append(item)
                            receptions.append(env.now)
                            latencies.append(env.now - item)
                    yield from update_entries(items)

                simulation_log.time.append(env.now)
                simulation_log.item_count.append(len(items))
                simulation_log.timing.append(env.now - start_time)
                simulation_log.mean_latency.append(np.mean(latencies) if latencies else 0)
                simulation_log.max_latency.append(np.max(latencies) if latencies else 0)
                simulation_log.message_emissions.extend(emissions)
                simulation_log.message_receptions.extend(receptions)
                simulation_log.message_latencies.extend(latencies)

                yield from wait_for_polling(start_time)

        env.process(self.arrival_strategy(env, db_lock, item_queue))
        env.process(consumer())
        env.run(until=until)
        return simulation_log


def performance_test():
    real_data = pd.read_csv('trades200.csv', sep=';', parse_dates=['time'],
                            date_parser=lambda x: pd.datetime.strptime(x, '%H:%M:%S.%f'))

    input_distributions = [
        # UniformItemArrival(5),
        # BurstyItemArrival(5, 10),
        RealItemArrival(real_data['time'])
    ]

    chunk_reports = []
    message_reports = []
    for input_distribution in input_distributions:
        simulation = PerformanceTest(arrival_strategy=input_distribution)
        result = simulation.run(until=input_distribution.total_duration)
        chunk_reports.append(result.to_chunk_report())
        message_reports.append(result.to_message_report())

    pyplot.subplot(5, 1, 1)
    for result in chunk_reports:
        pyplot.plot(result['time'], result['timing'])
        # pyplot.plot(result['time'], result['item_count'])
    pyplot.ylabel('Timing')

    pyplot.subplot(5, 1, 2)
    for result in chunk_reports:
        pyplot.plot(result['time'], result['mean_latency'])
        pyplot.plot(result['time'], result['max_latency'])
    pyplot.ylabel('Latency')

    pyplot.subplot(5, 1, 3)
    for result in message_reports:
        pyplot.plot_date(result['emission'], result['latency'], alpha=0.5, markersize=1)
        # pyplot.plot_date(result['reception'], result['latency'], alpha=0.5, markersize=1)
        # pyplot.scatter(result['reception'], result['latency'], alpha=0.5, s=1)
    pyplot.ylabel('Message latency')

    # '''
    pyplot.subplot(5, 1, 4)
    pyplot.plot_date(real_data['time'], real_data['latency'], alpha=0.5, markersize=1)
    pyplot.xlabel('Latency / Arrival rate')

    # Display the density of X over time (show the clustering happening)
    pyplot.subplot(5, 1, 5)
    resampled_data = real_data.set_index('time').resample('50ms').count()
    print(resampled_data)
    pyplot.plot_date(resampled_data.index, resampled_data['latency'], alpha=0.5, markersize=1, linestyle='-')
    pyplot.xlabel('Latency / Arrival rate')
    # '''

    '''
    pyplot.subplot(5, 1, 5)
    pyplot.plot(1000 / input_inter_arrivals, [result['mean_latency'].mean() for result in chunk_reports])
    pyplot.plot(1000 / input_inter_arrivals, [result['max_latency'].max() for result in chunk_reports])
    pyplot.xlabel('Latency / Arrival rate')
    '''

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
