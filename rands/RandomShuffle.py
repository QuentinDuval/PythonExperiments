from typing import *

import random


def shuffle(nums: List[Any]):
    """
    The idea is to generate a random permutation of the elements

    To generate a random permutation, the obvious way is to:
    - Pick an index from 0 to N - 1 and put it as first place
    - Pick an index from the remaining set of indices and put it as second place
    - etc

    This is completely equivalent to:
    - Pick an index from 0 to N - 1 and swap with index 0
    - Pick an index from 1 to N - 1 and swap with index 1
    - etc
    """
    for i in range(len(nums)):
        j = random.randint(i, len(nums)-1)
        nums[i], nums[j] = nums[j], nums[i]


xs = list(range(1, 50))
shuffle(xs)
print(xs)

