"""
https://leetcode.com/problems/remove-invalid-parentheses/

Remove the minimum number of invalid parentheses in order to make the input string valid. Return all possible results.

Note: The input string may contain letters other than the parentheses ( and ).
"""


from functools import lru_cache
from typing import List


class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        """
        A kind of dynamic programming approach:

        Recurrence:

        IP(start=i, opened=k)
        * if s[i] == '(', try both:
            * IP(i+1, k+1) - keep it
            * IP(i+1, k)   - ignore it
        * if s[i] == ')', try both:
            * IP(i+1, k-1)  - keep it (only if k > 0)
            * IP(i+1, k)    - ignore it

        End condition:

        IP(start=len(s), openend=k) => only works if k == 0

        Overlapping solutions for sure...
        O(N**2) sub-problems here

        The solutions will not be unique...
        (two parenthesis in a row lead to same result if remove one of them)
        => Unicity is easily doable by passing the last character read (not kept) as input

        This will help you count the number of solutions, but not have them, and not filter them
        by least removal.
        => You need to keep a count of the removed indices and keep the one with lowest...
        """

        @lru_cache(maxsize=None)
        def recur(pos: int, opened: int) -> List[List[int]]:
            if opened < 0:
                return []
            if pos == len(s):
                return [[]] if not opened else []

            if s[pos] == '(':
                return recur(pos + 1, opened + 1) + [[pos] + sol for sol in recur(pos + 1, opened)]
            elif s[pos] == ')':
                return recur(pos + 1, opened - 1) + [[pos] + sol for sol in recur(pos + 1, opened)]
            else:
                return recur(pos + 1, opened)

        def remove_indexes(indexes):
            indexes = set(indexes)
            out = ""
            for i, c in enumerate(s):
                if i not in indexes:
                    out += c
            return out

        solutions = set()
        valid_removals = recur(0, 0)
        min_removal = min(len(x) for x in valid_removals)
        for sol in valid_removals:
            if len(sol) == min_removal:
                solutions.add(remove_indexes(sol))
        return list(solutions)

        # TODO - Could do better by DFS:
        # https://leetcode.com/problems/remove-invalid-parentheses/discuss/340390/24ms-Python-dfs-Solution-beats-95
