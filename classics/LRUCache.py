from collections import *
import functools
import numpy as np

from Decorators import *


@traced
def read_resource(resource_name):
    with open(resource_name, 'r') as f:
        return len(f.read())


@functools.lru_cache(maxsize=4)
def long_computation(arg):
    print("Long computation", arg)
    return read_resource(arg)


class LRUCache:
    def __init__(self, max_size):
        self.dict = OrderedDict()
        self.max_size = max_size

    def __len__(self):
        return len(self.dict)

    def __contains__(self, key):
        if key in self.dict:
            self.dict.move_to_end(key, last=False)  # Move to beginning
            return True
        return False

    def __setitem__(self, key, value):
        self.dict[key] = value
        self.dict.move_to_end(key, last=False)      # Move to beginning
        if len(self) > self.max_size:
            self.dict.popitem(last=True)            # Pop the last key

    def __getitem__(self, key):
        val = self.dict.get(key)
        if val is not None:
            self.dict.move_to_end(key, last=False)  # Move to beginning
        return val


'''
class LRUCache(OrderedDict):
    def __init__(self, max_size):
        super().__init__()
        self.max_size = max_size

    def __contains__(self, key):
        try:
            super().move_to_end(key, last=False)
            return True
        except KeyError:
            return False

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().move_to_end(key, last=True)
        if len(self) > self.max_size:
            super().popitem(last=False)

    def __getitem__(self, key):
        val = super().get(key)
        # print(self.dict, key, val)
        if val is not None:
            super().move_to_end(key, last=False)
        return val
'''


def test_lru_cache():
    resources = ["Fibo.py", "IntegerBreak.py", "NLPFastText.py", "Simulation.py", "Simulation2.py"] * 10
    np.random.shuffle(resources)

    # resources = ['Simulation2.py', 'NLPFastText.py', 'Fibo.py', 'IntegerBreak.py', 'Simulation.py', 'Simulation2.py', 'Fibo.py', 'Fibo.py', 'IntegerBreak.py', 'Fibo.py', 'NLPFastText.py', 'Simulation2.py', 'NLPFastText.py', 'NLPFastText.py', 'Simulation.py', 'IntegerBreak.py', 'Simulation.py', 'Simulation2.py', 'IntegerBreak.py', 'Simulation.py']
    print(resources)

    print("-" * 50)

    for resource in resources:
        read_resource_cached(resource)

    print("-" * 50)

    cache = LRUCache(max_size=4)
    for resource in resources:
        if resource not in cache:
            cache[resource] = read_resource(resource)
        else:
            # print("HIT", resource)
            pass


test_lru_cache()





