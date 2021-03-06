"""
https://leetcode.com/problems/frog-jump/

A frog is crossing a river. The frog can jump on a stone, but it must not jump into the water.

Given a list of stones' positions in sorted ascending order, determine if the frog is able to cross the river
by landing on the last stone.

Initially, the frog is on the first stone and assume the first jump must be 1 unit.

If the frog's last jump was k units, then its next jump must be either k - 1, k, or k + 1 units.

Note that the frog can only jump in the forward direction.
"""


from typing import List


"""
First solution based on Dynamic Programming:
- try all possible jumps: last_jump - 1 .. last_jump + 1
- try the longest jump first (significantly speed up things)

Number of sub-problems:
- N position
- sqrt(N) (cause we can increment by one, sum of 1 to N is N ** 2)
=> O(N ** 3/2)

Time complexity is O(N ** 3/2 * log N) - the log N is due to binary search
Space complexity is O(N ** 3/2)

Beats 62% (200ms)
"""


class Solution:
    def canCross(self, stones: List[int]) -> bool:
        if not stones:
            return True

        def cache(f):
            memo = {}

            def wrapped(*args):
                res = memo.get(args)
                if res is not None:
                    return res
                res = f(*args)
                memo[args] = res
                return res

            return wrapped

        def b_search(lo: int, pos: int) -> int:
            hi = len(stones) - 1
            while lo <= hi:
                mid = lo + (hi - lo) // 2
                if pos < stones[mid]:
                    hi = mid - 1
                elif pos > stones[mid]:
                    lo = mid + 1
                else:
                    return mid
            return None

        @cache
        def visit(i: int, last_jump: int) -> bool:
            if i == len(stones) - 1:
                return True

            for jump in range(max(1, last_jump - 1), last_jump + 2):
                j = b_search(i, stones[i] + jump)
                if j is not None and visit(j, jump):
                    return True
            return False

        return visit(0, 0)


"""
We can improve efficiency by storing positions in a set:
- we do not need to know the exact position in the list
- we only need to know if we reached the end

Time complexity decreased to O(N ** 3/2)

Beats 86% (88%)
"""


class Solution:
    def canCross(self, stones: List[int]) -> bool:
        if not stones:
            return True

        start_pos = stones[0]
        last_pos = stones[-1]
        stones = set(stones)

        def cache(f):
            memo = {}

            def wrapped(*args):
                res = memo.get(args)
                if res is not None:
                    return res
                res = f(*args)
                memo[args] = res
                return res

            return wrapped

        @cache
        def visit(pos: int, last_jump: int) -> bool:
            if pos == last_pos:
                return True

            for jump in reversed(range(max(1, last_jump - 1), last_jump + 2)):
                if pos + jump in stones and visit(pos + jump, jump):
                    return True
            return False

        return visit(start_pos, 0)


"""
Finally, we can reason that we do not really need the intermediary result of DP:
- just do a DFS
- keep a 'discovered' set to avoid visiting the same (position, last_jump) twice

Complexity stays the same
Beats 89% (76 ms)
"""


class Solution:
    def canCross(self, stones: List[int]) -> bool:
        if not stones:
            return True

        start_pos = stones[0]
        last_pos = stones[-1]
        stones = set(stones)

        discovered = {(start_pos, 0)}
        to_visit = [(start_pos, 0)]

        while to_visit:
            pos, last_jump = to_visit.pop()
            for jump in range(max(1, last_jump - 1), last_jump + 2):
                destination = pos + jump
                if destination == last_pos:
                    return True

                if destination not in stones:
                    continue

                if (destination, jump) not in discovered:
                    to_visit.append((destination, jump))
                    discovered.add((destination, jump))

        return False
