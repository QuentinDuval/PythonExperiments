from collections import *
from typing import *


"""
Finding the median by divide and conquer
"""


# TODO


"""
Finding the missing number from a list of N-1 number that should hold the numbers from 1 to N
"""


# TODO


"""
Find the number replicated several times in a list of number holding otherwise unique numbers
"""


def find_duplicate(nums: List[int]) -> int:
    """
    Proof:
    Pigeon hole principle states that there must be one hole with 2 pigeons

    Find by Divide and conquer:
    - Take the average value of min(nums) and max(nums)
    - Count the number of elements lower than this median
    - Count the number of elements higher than this median
    - Recurse in the bigger half

    Complexity is O(n * log n)
    """

    def count(lo, mid, hi):
        l_count = r_count = m_count = 0
        for n in nums:
            if lo <= n and n <= hi:
                if n < mid:
                    l_count += 1
                elif n > mid:
                    r_count += 1
                else:
                    m_count += 1
        return l_count, m_count, r_count

    lo = 1
    hi = len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        l_count, m_count, r_count = count(lo, mid, hi)
        if m_count > 1:
            return mid
        elif l_count > mid - lo:
            hi = mid - 1
        else:
            lo = mid + 1


"""
Find the longest contiguous sub-sequence that contains at least K of each of the different the letters it contains
"""


# TODO


"""
Design a calendar: https://leetcode.com/problems/my-calendar-i/
"""


Slot = namedtuple('Slot', ['start', 'end'])


class MyCalendar:
    def __init__(self):
        self.slots = []

    def book(self, start: int, end: int) -> bool:
        slot = Slot(start, end)
        i = self.lower_bound(start)
        if i < len(self.slots):
            if self.slots[i].start < end:
                return False

        if i > 0:
            if self.slots[i - 1].end > start:
                return False

        self.slots.insert(i, slot)
        return True

    def lower_bound(self, start):
        lo = 0
        hi = len(self.slots) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if start < self.slots[mid].start:
                hi = mid - 1
            else:
                lo = mid + 1
        return lo
