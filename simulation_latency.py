"""
Simulation of latency of servers:
- simulation if you have only concurrent request
- simulation if you have join requests (one request after the other)

Articles:
https://realpython.com/python-histograms/
"""

from typing import List

import dataclasses
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclasses.dataclass()
class Service:
    processing_time: float
    dependencies: List[str]


@dataclasses.dataclass()
class Parameters:
    iter_count: int
    default_processing_time: float
    network_latency: float
    sequential_requests: bool


services = {
    "a": Service(processing_time=5, dependencies=["b", "c"]),
    "b": Service(processing_time=5, dependencies=["d", "e", "f", "g"]),
    "c": Service(processing_time=5, dependencies=["h", "i"]),
    # "d": Service(processing_time=1, dependencies=["j", "k", "l", "m"]),
    # "e": Service(processing_time=1, dependencies=["n", "o"]),
}


def compute_delay(services, start, params: Parameters):
    def recur(node):
        service = services.get(node)
        if not service:
            return np.random.exponential(params.default_processing_time, 1)
        spent = service.processing_time + np.random.exponential(service.processing_time * 0.2, 1)
        reducer = sum if params.sequential_requests else max
        return spent + reducer(recur(d) + params.network_latency + np.random.exponential(params.network_latency * 0.2, 1) for d in service.dependencies)
    return recur(start)


def expected_delay(services, start, params: Parameters):
    delays = [compute_delay(services, start, params) for _ in range(params.iter_count)]
    df = pd.DataFrame(data=delays, columns=["delays"])
    print(df.describe())
    df.plot.hist(bins=1 + int(max(delays)), alpha=0.5)
    plt.title('For ' + str(params.iter_count) + ' Tests')
    plt.xlabel('Counts')
    plt.ylabel('Total, delay')
    plt.grid(axis='y', alpha=.75)
    plt.show()


expected_delay(services, start="a", params=Parameters(default_processing_time=5,
                                                      iter_count=10_000,
                                                      network_latency=1,
                                                      sequential_requests=False))


# TODO - show the expected delay by node (waiting time) and the processing time
# TODO - show the graph of the services
# TODO - you could have a time that depends on the number of times a service is being used (to simulate overload)
