"""
https://leetcode.com/problems/boats-to-save-people/

The i-th person has weight people[i], and each boat can carry a maximum weight of limit.

Each boat carries at most 2 people at the same time, provided the sum of the weight of those people is at most limit.

Return the minimum number of boats to carry every given person.  (It is guaranteed each person can be carried by a boat.)
"""


from typing import List


class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        """
        Greedy:
        - pair the heaviest with the lighter person if possible
        - else put the heaviest alone

        This works because a boat can carry at most 2 people, and so a lighter person
        cannot be blocked by a heavier person taking the lighest pair.

        Complexity is O(N log N)
        Beats 75%
        """
        people.sort()
        count = 0
        lo = 0
        hi = len(people) - 1
        while lo <= hi:
            count += 1
            if people[hi] + people[lo] <= limit:
                lo += 1
            hi -= 1
        return count
