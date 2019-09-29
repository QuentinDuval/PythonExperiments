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
                # yield env.timeout(np.random.exponential(self.processing_time))
                served_time = env.now
                times.append(served_time - sent_time)

        env.process(request_arrival())
        for _ in range(self.nb_threads):
            env.process(server_thread())
        env.run(until=until)
        return times


def test():
    simulations = [
        ServerSimulation(processing_time=0.1, nb_threads=200).run(schedule=[1000, 1000 * 3, 1000, 1000, 1000], until=10),
        ServerSimulation(processing_time=0.1, nb_threads=200).run(schedule=[1250, 1250 * 3, 1250, 1250, 1250], until=10),
        ServerSimulation(processing_time=0.1, nb_threads=200).run(schedule=[1500, 1500 * 3, 1500, 1500, 1500], until=10),
        ServerSimulation(processing_time=0.1, nb_threads=200).run(schedule=[1750, 1750 * 3, 1750, 1750, 1750], until=10),
        ServerSimulation(processing_time=0.1, nb_threads=200).run(schedule=[2000, 2000 * 3, 2000, 2000, 2000], until=10),
        ServerSimulation(processing_time=0.1, nb_threads=200).run(schedule=[2250, 2250 * 3, 2250, 2250, 2250], until=10),
    ]

    fig, ax = plot.subplots(nrows=len(simulations), ncols=1, sharex='all')

    for i, sim in enumerate(simulations):
        df = pd.DataFrame(data=sim, columns=['latency'])
        print(df.describe(percentiles=[0.1, 0.5, 0.75, 0.9, 0.95]))
        ax[i].hist(x=sim, bins=20, density=True)

    plot.show()


test()
