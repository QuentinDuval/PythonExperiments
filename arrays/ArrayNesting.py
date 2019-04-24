from typing import List


class Solution:
    def arrayNesting(self, nums: List[int]) -> int:
        """
        We are basically looking for 'cycles'

        We could try every cycles at each positions, but that would be stupid:
        We do not have to test for a position already visited (through another cycle)

        Why does it work?
        - The algorithm below would not work if a line could join a cycle.
        - But we can only have cycles for numbers cannot repeat!

        Complexity: O(N) time and O(N) space
        """
        n = len(nums)

        count = 0
        visited = set()

        for i in range(n):
            if i in visited:
                continue

            cycle_len = 0
            while i not in visited:
                visited.add(i)
                i = nums[i]
                cycle_len += 1
            count = max(count, cycle_len)

        return count
