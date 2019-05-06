"""
Form with TWO NUMBERS:

Given an array A of N elements.
Complete the function which returns true if 4-sum exists in array A whose sum is target else return false.
"""


from collections import defaultdict
from typing import List


def find_four_sums(nums: List[int], target: int) -> List[List[int]]:
    """
    We can reduce to the problem of 2-sums by pre-computing the sum of all pairs,
    putting these sum of pairs in a hash table, but then we need to take care of
    not having duplicates.

    One way to do this is to do a two-passes algorithm:
    - Fill a hash map with the sum of all pairs to their indices
    - Visit the hash map, looking for other pairs with partial sum = target - current partial sum
    Beware, there might be duplications, and there might be many such pairs.

    Another way is to do a one pass algorithm:
    - Fill the hash-map AFTER having visited a value (add the N partial sum with new index)
    - Search in the hash map for all pairs after the current value
    => This automatically takes care of duplicates
    """
    if len(nums) < 4:
        return []

    combinations = []
    visited = defaultdict(list)
    visited[nums[0] + nums[1]].append((0, 1))

    for c in range(2, len(nums) - 1):
        for d in range(c + 1, len(nums)):
            partial = nums[c] + nums[d]
            for a, b in visited.get(target - partial, []):
                combinations.append((nums[a], nums[b], nums[c], nums[d]))

        # complete the hash map
        for i in range(0, c):
            visited[nums[i] + nums[c]].append((i, c))

    return combinations


print(find_four_sums([1, 0, -1, 0, -2, 2], 0))
