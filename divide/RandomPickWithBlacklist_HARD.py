"""
https://leetcode.com/problems/random-pick-with-blacklist

Given a blacklist B containing unique integers from [0, N), write a function to return a uniform random
integer from [0, N) which is NOT in B.

Optimize it such that it minimizes the call to system’s Math.random().

Note:
* 1 <= N <= 1000000000
* 0 <= B.length < min(100000, N)
"""

from typing import List
import random


class Solution:
    """
    The brute force would be to try to pick a random number and try again and again
    if it falls inside the blacklist but this is very slow.

    Another approach would be to create a list [0, N) minus the blacklist.
    Then we could pick an index inside this list. But it consumes too much memory.

    The next approach is instead to pick a number between 0 and N - len(blacklist)
    then deduce by how much we must increase it to have the real number.

    Basically:
    - we pick a value X in [0, N - len(blacklist))
    - we search D such that if we search X + D inside the blacklist there are D numbers befores
    - then we report X + D as output

    One approach is to search X inside blacklist, then count how many we skipped D1, then
    search X + D1, count if we skipped more D2, then search X + D1 + D2... until we do not move
    then report this number.
    => The problem is that it is O(len(blacklist)) at worse (think consecutive numbers)

    Another approach is to binary search directly inside the blacklist.
    * if LEFT_COUNT + X < blacklist[LEFT_COUNT], we know we should search left
    * else we should search right (equal means we should search right as well)

    The smallest number LEFT_COUNT such that LEFT_COUNT + X > blacklist[LEFT_COUNT]
    is the one we search: it is a lower bound search.

    Example:
    * N = 10
    * B = [2, 5, 6, 8]

    We pick 4 in [0, 10 - len(B))

    First search
    [2, 5, 6, 8]
        ^

    mid is 5: 1 + 4 >= 5 => go right (lo = 2)
    mid is 6: 2 + 4 >= 6 => go right (lo = 3)
    mid is 8: 3 + 4 < 8  => go left  (hi = 2)

    Report lo + 4 = 7
    """

    def __init__(self, n: int, blacklist: List[int]):
        self.blacklist = blacklist
        self.blacklist.sort()
        self.max_pick = n - 1 - len(self.blacklist)

    def pick(self) -> int:
        searched = random.randint(0, self.max_pick)
        lo = 0
        hi = len(self.blacklist) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            val_mid = self.blacklist[mid]
            if mid + searched < val_mid:
                hi = mid - 1
            else:
                lo = mid + 1
        return searched + lo

