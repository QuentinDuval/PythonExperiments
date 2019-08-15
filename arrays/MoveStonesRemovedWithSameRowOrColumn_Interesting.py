"""
https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/

On a 2D plane, we place stones at some integer coordinate points.  Each coordinate point may have at most one stone.

Now, a move consists of removing a stone that shares a column or row with another stone on the grid.

What is the largest possible number of moves we can make?
"""

from typing import *


class Solution:
    def removeStones(self, stones: List[List[int]]) -> int:
        """
        One could try to group the stones by 'x' and 'y' and try systematically
        to remove each stones in order:
        - if already occupied, remove the stone
        - if still a conflict, try both removing the stone or not
        - if no conflict, you must keep the stone

        But this is wrong... it does not even work.
        There is NO ORDER in the solution => NO DP POSSIBLE
        In any case, it would explore too many solutions.
        """

        '''
        sharing_x = defaultdict(set)
        sharing_y = defaultdict(set)
        for i, (x, y) in enumerate(stones):
            sharing_x[x].add(i)
            sharing_y[y].add(i)

        def can_remove(x, y):
            return len(sharing_x[x]) > 1 or len(sharing_y[y]) > 1

        def visit(rows: Set[int], cols: Set[int], pos: int) -> int:
            if pos == len(stones):
                return 0

            cost = 0
            x, y = stones[pos]
            if x in rows or y in cols or can_remove(x, y):
                sharing_x[x].remove(pos)
                sharing_y[y].remove(pos)
                cost = 1 + visit(rows, cols, pos+1)
                sharing_x[x].add(pos)
                sharing_y[y].add(pos)

            if x not in rows and y not in cols:
                rows.add(x)
                cols.add(y)
                cost = max(cost, visit(rows, cols, pos+1))
                rows.remove(x)
                cols.remove(y)
            return cost

        return visit(set(), set(), 0)
        '''

        """
        In fact, if you draw a diagram, you will see it can be seen as a graph:
        - points that share the same X or Y are linked
        - points that are linked can be intelligently be removed, keeping only 1
        => So we just need to count the number of connected components

        We can create a graph where a node 'i' is connected to its 'x' and 'y'.

        But we can do something better based on UNION-FIND:
        - union the points that share the same 'x'
        - union the points that share the same 'y'
        => group the points by 'x' and also by 'y'
        
        Beats 99.21%
        """

        n = len(stones)
        parents = list(range(n))
        ranks = [0] * n

        def find(i):
            while i != parents[i]:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        def union(i, j):
            x = find(i)
            y = find(j)
            if ranks[x] > ranks[y]:
                parents[y] = x
                ranks[x] += ranks[y]
            else:
                parents[x] = y
                ranks[y] += ranks[x]

        sharing_x = {}
        sharing_y = {}
        for i, (x, y) in enumerate(stones):
            if x in sharing_x:
                union(sharing_x[x], i)
            else:
                sharing_x[x] = i
            if y in sharing_y:
                union(sharing_y[y], i)
            else:
                sharing_y[y] = i

        number_cc = len(set(find(i) for i in range(n)))
        return n - number_cc
