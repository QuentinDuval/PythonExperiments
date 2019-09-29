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


def simulate(avg_percentage, spike_multiplier):
    proc_time = 1
    nb_threads = 10
    max_throughput = nb_threads / proc_time
    avg = int(max_throughput * avg_percentage)
    spike = int(avg * spike_multiplier)
    return avg_percentage, ServerSimulation(processing_time=proc_time, nb_threads=nb_threads).run(
        schedule=[avg, spike, avg, avg, spike, avg, avg, spike, avg, avg] + [avg] * 10,
        until=50)


def test():
    spike = 1.5
    simulations = [
        simulate(0.5, spike),
        simulate(0.6, spike),
        simulate(0.7, spike),
        simulate(0.8, spike),
        simulate(0.9, spike),
        simulate(1.0, spike),
    ]

    dfs = []
    for i, (avg, sim) in enumerate(simulations):
        dfs.append(pd.DataFrame(data=sim, columns=[str(avg)]))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    df = pd.concat(dfs, sort=False)
    print(df.describe(percentiles=[0.1, 0.5, 0.75, 0.9, 0.95]))

    # '''
    fig, ax = plot.subplots()
    for i, (avg, sim) in enumerate(simulations):
        ax.hist(x=sim, bins=100, range=(0.75, 3), density=True)
    plot.show()
    # '''

    '''
    fig, ax = plot.subplots(nrows=len(simulations), ncols=1, sharex='all')

    for i, sim in enumerate(simulations):
        df = pd.DataFrame(data=sim, columns=['latency'])
        print(df.describe(percentiles=[0.1, 0.5, 0.75, 0.9, 0.95]))
        ax[i].hist(x=sim, bins=100, density=True)

    plot.show()
    '''


test()
