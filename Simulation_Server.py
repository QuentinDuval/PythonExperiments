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
                # yield env.timeout(self.processing_time)
                # yield env.timeout(np.random.exponential(self.processing_time))
                yield env.timeout(np.random.normal(self.processing_time, self.processing_time * 0.1))
                served_time = env.now
                times.append(served_time - sent_time)

        env.process(request_arrival())
        for _ in range(self.nb_threads):
            env.process(server_thread())
        env.run(until=until)
        return times


def simulate(avg_percentage, spike_multiplier):
    # TODO - instead of this logic with SPIKE, do a standard distribution ? not really... TREND modeling
    # TODO - add a logic to do the retry past a given threshold...

    proc_time = 0.2
    nb_threads = 200
    max_throughput = nb_threads / proc_time
    avg = max_throughput * avg_percentage
    spike = int(avg * spike_multiplier)

    nb_period = 30
    nb_spike = 3
    normal = int((nb_period * avg - spike * nb_spike) / (nb_period - nb_spike))
    schedule = [normal] * 10 + [normal, spike, normal, normal, spike, normal, normal, spike, normal, normal] + [normal] * 10
    return avg_percentage, ServerSimulation(processing_time=proc_time, nb_threads=nb_threads).run(until=nb_period * 2, schedule=schedule)


def test():
    spike = 2.0
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
    summary = df.describe(percentiles=[0.1, 0.5, 0.75, 0.9, 0.95])
    print(summary)

    '''
    fig, ax = plot.subplots()
    for i, (avg, sim) in enumerate(simulations):
        ax.hist(x=sim, bins=100, range=(0., 5), density=True)
    plot.show()
    '''

    # '''
    # fig, ax = plot.subplots(nrows=len(simulations) // 2, ncols=2, sharey='all')
    fig, ax = plot.subplots(nrows=len(simulations) // 2, ncols=2)
    for i, (avg, sim) in enumerate(simulations):
        graph_coord = (i // 2, i % 2)
        ax[graph_coord].set_title(str(avg * 100))
        ax[graph_coord].hist(x=sim, bins=400, range=(0., 3), density=True)
    plot.tight_layout()
    plot.show()
    # '''

    fig, ax = plot.subplots(nrows=2, ncols=2)
    for i, name in enumerate(['mean', '50%', '75%', '95%']):
        graph_coord = (i // 2, i % 2)
        ax[graph_coord].set_title(name)
        ax[graph_coord].plot(summary.loc[name, :])
    plot.tight_layout()
    plot.show()


test()
