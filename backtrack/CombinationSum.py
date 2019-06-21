"""
https://leetcode.com/problems/combination-sum/

Given a set of candidate numbers (candidates) (without duplicates) and a target number (target), find all unique
combinations in candidates where the candidate numbers sums to target.

The same repeated number may be chosen from candidates unlimited number of times.

Note:
* All numbers (including target) will be positive integers.
* The solution set must not contain duplicate combinations.
"""


from typing import List


class Solution:
    def combinationSum_1(self, nums: List[int], target: int) -> List[List[int]]:
        """
        If we select the number at 'i', then we have another sub-problem to solve at 'i+1' with 'target-nums[i]'.
        If we sort the numbers, we can cut the exploration when nums[i] > target.

        There might be overlapping problems (select 2 then 3, or select 5), but it might not be worth
        memoizing them because we systematically copy the solutions (worth it if copy is is less).
        """

        def backtrack(pos: int, target: int) -> List[List[int]]:
            if target == 0:
                return [[]]

            if pos >= len(nums) or nums[pos] > target:
                return []

            return [[nums[pos]] + sub_sol for sub_sol in backtrack(pos, target - nums[pos])] + backtrack(pos + 1,
                                                                                                         target)

        nums.sort()
        return backtrack(0, target)

    def combinationSum(self, nums: List[int], target: int) -> List[List[int]]:
        """
        We can turn the recusion differently:
        - Use division to try all possible number of time you can put nums[i]
        - Use iteration to select the next num once you selected one (or zero)

        Additional tricks
        - Use 'yield' to perform a lazy evaluation
        - Use a path to avoid copy of lists
        """

        solutions = []

        def backtrack(pos: int, target: int, path: List[int]) -> List[List[int]]:
            if target == 0:
                yield list(path)
                return

            if pos >= len(nums) or nums[pos] > target:
                return

            q, r = divmod(target, nums[pos])
            path.extend([nums[pos]] * q)

            yield from backtrack(pos + 1, r, path)
            for i in range(q):
                r += nums[pos]
                path.pop()
                yield from backtrack(pos + 1, r, path)

        nums.sort()
        return list(backtrack(0, target, path=[]))
