"""
Simulation of latency of servers:
- simulation if you have only concurrent request
- simulation if you have join requests (one request after the other)
"""

import numpy as np
import pandas as pd


services = {
    "a": ["b", "c"],
    "b": ["d", "e", "f", "g"],
    "c": ["i", "j"]
}


def compute_delay(services, start):
    def recur(node):
        children = services.get(node, [])
        if not children:
            return np.random.exponential(5, 1)
        return max(recur(child) for child in children)
    return recur(start)


def expected_delay(services, start, iter_count):
    # TODO - show the expected delay by node (waiting time) and the processing time
    delays = [compute_delay(services, start) for _ in range(iter_count)]
    df = pd.DataFrame(data=delays, columns=["delays"])
    print(df.describe())


expected_delay(services, "a", 100)
