"""
https://leetcode.com/problems/array-of-doubled-pairs/

Given an array of integers A with even length, return true if and only if it is possible to reorder it such
that A[2 * i + 1] = 2 * A[2 * i] for every 0 <= i < len(A) / 2.
"""


from typing import List


class Solution:
    def canReorderDoubled(self, nums: List[int]) -> bool:
        """
        Brute force algorithm would be for every number to search a doubling number
        - keep track of which number (just an array of boolean on the side) are matched
        - skip numbers for which you cannot find a double
          (do not search for half or double, it becomes a search problem)
        - do a second pass on which numbers are not matched
        => Complexity would be O(N**2)

        We can accelerate the search for the double number by storing each number in an
        hash map along with a counter:
        - idem: skip numbers for which you cannot find a double
        - remove nums for which the count reached 0
        - the second pass is just "is my map empty?"
        => Complexity drops to O(N)

        Unfortunately, it does not work:
        You need to sort the numbers and deal with the number in that order
        => Complexity is O(N log N)

        You can greatly improve the performance by searching by group of numbers
        """

        counter = {}
        for num in nums:
            counter[num] = counter.get(num, 0) + 1

        for num in sorted(counter.keys(), key=abs):
            count = counter.get(num, 0)
            if num == 0:
                if count % 2 == 1:
                    return False
                counter[num] = 0
            elif count == 0:
                continue
            else:
                double_count = counter.get(num * 2, 0)
                if double_count < count:
                    return False

                counter[num] = 0
                counter[num * 2] -= count

        return True


