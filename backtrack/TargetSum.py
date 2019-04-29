from typing import List


class Solution:
    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        """
        For each number, you must either add it or subtract it, and the goal is to find the target S.

        Based on the length of the array (maximum 20) we can do:
        - Backtracking (exhaustive search)
        - With pruning to speed it up: the target must be in the range [-sum nums, +sum nums]
        - With memoization (since we might have same input visited)

        Number of sub-problems? (2 * MaxSum + 1) * N

        WHEN YOU DO RECURSION ALWAYS THINK ABOUT MEMOIZATION
        """
        memo = {}

        def visit(i: int, total: int, target: int) -> int:
            result = memo.get((i, target))
            if result is not None:
                return result

            if i == len(nums):
                return 1 if target == 0 else 0

            if target < -total or total < target:
                return 0

            next_total = total - abs(nums[i])
            result = visit(i + 1, next_total, target - nums[i]) + visit(i + 1, next_total, target + nums[i])
            memo[(i, target)] = result
            return result

        return visit(0, total=sum(abs(n) for n in nums), target=S)
