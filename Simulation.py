import collections
import functools
import numpy as np
import heapq

TaxiEvent = collections.namedtuple(
    'TaxiEvent', ['time', 'taxi_id', 'description']
)

class TaxiCab:
    def __init__(self, taxi_id):
        self.id = taxi_id

    def new_simulation(self, start_time, end_time):
        """
        New passengers will be accepted up till 'end_time' at which point the taxi goes back home
        """
        time = yield TaxiEvent(taxi_id=self.id, description='start working day', time=start_time)
        while time < end_time:
            time = yield TaxiEvent(taxi_id=self.id, description='take passenger', time=time)
            time = yield TaxiEvent(taxi_id=self.id, description='drop passenger', time=time)
        yield TaxiEvent(time=time, taxi_id=self.id, description='end of working day')


def simulate(taxis, start_time, end_time, trip_duration):
    pending_events = []
    simulations = {taxi.id: taxi.new_simulation(start_time, end_time) for taxi in taxis}
    for simulation in simulations.values():
        heapq.heappush(pending_events, next(simulation))

    while pending_events:
        event = heapq.heappop(pending_events)
        yield event
        try:
            simulation = simulations[event.taxi_id]
            duration = np.random.exponential(scale=trip_duration)
            event = simulation.send(event.time + duration)
            heapq.heappush(pending_events, event)
        except StopIteration:
            del simulations[event.taxi_id]


def test(taxi_count):
    taxis = [TaxiCab(taxi_id=i+1) for i in range(taxi_count)]
    for e in simulate(taxis, start_time=0, end_time=10, trip_duration=0.5): # TODO - different waiting times for different events
        print(e)


# TODO - take into account money
# TODO - count how late taxi end up their day
test(taxi_count=10)

