import collections
import enum
import functools
import heapq
import numpy as np


class TaxiEventType(enum.Enum):
    START_DAY = 0
    TAKE_PASSENGER = 1
    DROP_PASSENGER = 2
    END_DAY = 3


TaxiEvent = collections.namedtuple(
    'TaxiEvent', ['time', 'taxi_id', 'type']
)


class TaxiCab:
    def __init__(self, taxi_id):
        self.id = taxi_id

    def new_simulation(self, start_time, end_time):
        """
        New passengers will be accepted up till 'end_time' at which point the taxi goes back home
        """
        time = yield TaxiEvent(taxi_id=self.id, time=start_time, type=TaxiEventType.START_DAY)
        while time < end_time:
            time = yield TaxiEvent(taxi_id=self.id, time=time, type=TaxiEventType.TAKE_PASSENGER)
            time = yield TaxiEvent(taxi_id=self.id, time=time, type=TaxiEventType.DROP_PASSENGER)
        yield TaxiEvent(taxi_id=self.id, time=time, type=TaxiEventType.END_DAY)


class Simulation:
    def __init__(self, taxis, trip_duration, client_wait):
        self.taxis = taxis
        self.durations = {TaxiEventType.START_DAY: client_wait,
                          TaxiEventType.TAKE_PASSENGER: trip_duration,
                          TaxiEventType.DROP_PASSENGER: client_wait}

    def duration(self, event):
        expected = self.durations.get(event.type, 0)
        return np.random.exponential(scale=expected)

    def run(self, start_time, end_time):
        pending_events = []
        taxi_simulations = {taxi.id: taxi.new_simulation(start_time, end_time) for taxi in self.taxis}
        for taxi_simulation in taxi_simulations.values():
            heapq.heappush(pending_events, next(taxi_simulation))

        while pending_events:
            event = heapq.heappop(pending_events)
            yield event
            try:
                taxi_simulation = taxi_simulations[event.taxi_id]
                duration = np.random.exponential(scale=self.duration(event))
                event = taxi_simulation.send(event.time + duration)
                heapq.heappush(pending_events, event)
            except StopIteration:
                del taxi_simulations[event.taxi_id]


def test(taxi_count):
    simulation = Simulation(trip_duration=0.5, client_wait=0.25,
                            taxis=[TaxiCab(taxi_id=i+1) for i in range(taxi_count)])
    for e in simulation.run(start_time=0, end_time=10):
        print(e)


# TODO - take into account money
# TODO - count how late taxi end up their day
test(taxi_count=10)

