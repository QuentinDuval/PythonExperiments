"""
Goal of this simulation is to show at which point a server following a Spring Boot approach
will start drowning when faced with a burst of request:
- measure how many requests will be answered in time
- measure how much time is needed to go back to normal

Try with different number of threads.
=> Show that the more thread, the steeper the change (no warning basically)
"""


from typing import List

from collections import Counter
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import simpy


class ServerSimulation:
    def __init__(self, processing_time: float, nb_threads: int):
        self.processing_time = processing_time
        self.nb_threads = nb_threads

    def run(self, schedule: List[int], until=None) -> List[float]:
        # start = []
        times = []
        env = simpy.Environment()
        request_queue = simpy.Store(env)

        def request_arrival():
            for nb_requests in schedule:
                delay = 1.0 / nb_requests
                for _ in range(nb_requests):
                    request_queue.put(env.now)
                    yield env.timeout(delay)

        def server_thread():
            while True:
                sent_time = yield request_queue.get()
                yield env.timeout(self.processing_time)
                served_time = env.now
                times.append(served_time - sent_time)

        env.process(request_arrival())
        for _ in range(self.nb_threads):
            env.process(server_thread())
        env.run(until=until)
        return times


def test():
    simulation = ServerSimulation(processing_time=1, nb_threads=15)
    times = simulation.run(schedule=[10, 30, 10, 10], until=10)
    df = pd.DataFrame(data=times, columns=['delay'])
    print(df.describe())
    df.plot.hist()
    plot.show()


test()
