from collections import *
import functools
import numpy as np

from Decorators import *


@traced
def read_resource(resource_name):
    with open(resource_name, 'r') as f:
        return len(f.read())


@functools.lru_cache(maxsize=4)
def read_resource_cached(resource_name):
    return read_resource(resource_name)


class LRUCache(OrderedDict):
    def __init__(self, max_size):
        super(OrderedDict, self).__init__()
        self.max_size = max_size

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)
        # TODO - return something?


def test_lru_cache():
    resources = ["Fibo.py", "IntegerBreak.py", "NLPFastText.py", "Simulation.py", "Simulation2.py"] * 4
    np.random.shuffle(resources)

    for resource in resources:
        read_resource_cached(resource)

    print("-" * 50)

    cache = LRUCache(max_size=4)
    for resource in resources:
        if resource not in cache:
            cache[resource] = read_resource(resource)
        else:
            print("HIT", resource)


test_lru_cache()





