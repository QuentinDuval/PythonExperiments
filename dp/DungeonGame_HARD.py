"""
https://leetcode.com/problems/dungeon-game


"""


from typing import List

        

class Solution:
    def calculateMinimumHP(self, grid: List[List[int]]) -> int:
        """
        Solution 1:
        
        The idea is to explore the neighbors, always selecting the neighbors leading the highest current HP.
        - We stop when we reach the princess
        - We keep the minimum HP seen so far on the way
        
        Why does it work? Why does it gives us the minimum?
        Because we only visit a place if it is the highest possible path, so we can never go below the minimum.
        
        To make it work though, we have to limit the amount of paths visited:
        We have to use a replacement heap, that only keeps the 'highest HP' for a given position (x, y)

        Even with this, it does not work !!!
        => Timeout, cause we can visit the same place several times
        => If you try to add a 'visited', it fails (it is Dijsktra, but with negative weights...)
        """
        
        if not grid or not grid[0]:
            return 1
        
        h = len(grid)
        w = len(grid[0])
        princess = (h-1, w-1)
        
        def neighbors(x, y):
            if x < h - 1:
                yield x+1, y
            if y < w - 1:
                yield x, y+1
        
        min_hp = 0
        to_visit = MaxHeap()
        to_visit.push(grid[0][0], (0, 0))
        visited = set()
        
        while to_visit:
            hp, (x, y) = to_visit.pop()
            visited.add((x, y))
            
            min_hp = min(min_hp, hp)
            if (x, y) == princess:
                break
            
            for i, j in neighbors(x, y):
                if (i, j) not in visited:   # Without this, can visit same place several times
                    to_visit.push(hp + grid[i][j], (i, j))
        
        return 1 - min_hp


class MaxHeap:
    """
    Max Heap with replacement strategy to only keep the highest value
    """

    def __init__(self):
        self.position = {}
        self.heap = [(float('inf'), None)]  # i must be > 2 * i and 2 * i + 1

    def __len__(self):
        return len(self.position)

    def __repr__(self):
        return repr(self.heap)

    def push(self, priority, value):
        position = self.position.get(value)
        if position is not None:
            previous_priority = self.heap[position][0]
            if previous_priority < priority:
                self.heap[position] = (priority, value)
                self.swim_up(position)
        else:
            position = len(self.heap)
            self.position[value] = position
            self.heap.append((priority, value))
            self.swim_up(position)

    def pop(self):
        self.swap(1, -1)
        priority, value = self.heap.pop()
        del self.position[value]
        self.sink_down(1)
        return priority, value

    def swim_up(self, pos):
        father = pos // 2
        if self.heap[father][0] < self.heap[pos][0]:
            self.swap(father, pos)
            self.swim_up(father)

    def sink_down(self, pos):
        max_child = None
        if 2 * pos < len(self.heap):
            max_child = 2 * pos
        if 2 * pos + 1 < len(self.heap):
            if self.heap[2 * pos + 1][0] > self.heap[max_child][0]:
                max_child = 2 * pos + 1

        if max_child and self.heap[max_child][0] > self.heap[pos][0]:
            self.swap(max_child, pos)
            self.sink_down(max_child)

    def swap(self, pos1, pos2):
        self.heap[pos1], self.heap[pos2] = self.heap[pos2], self.heap[pos1]
        for pos in [pos1, pos2]:
            self.position[self.heap[pos][1]] = pos


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


class Solution:
    def calculateMinimumHP(self, grid: List[List[int]]) -> int:
        """
        Solution 2:
        The minimum HP can be obtained recursively (and then dynamic programming)

        If paths given a min HP of M1 and M2 recursively
        - take the best of M1 and M2 => max(M1, M2)
        - add the HP increase/decrease of current position
        - take the min of current position and max(M1, M2)

        Why? Because we want the minimum of the rest, plus some buffer added by
        current position. But if the current position is negative, it might
        also be the lowest point of the path to the princess.
        """

        if not grid or not grid[0]:
            return 1

        h = len(grid)
        w = len(grid[0])

        def neighbors(i, j):
            if i < h - 1:
                yield i + 1, j
            if j < w - 1:
                yield i, j + 1

        @cache
        def lowest_hp(i, j):
            if i == h - 1 and j == w - 1:
                return grid[i][j]

            lowest = max(lowest_hp(x, y) for x, y in neighbors(i, j))
            return min(grid[i][j], grid[i][j] + lowest)

        lo = lowest_hp(0, 0)
        if lo >= 0:
            return 1
        return -lo + 1


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


# TODO - bottom up DP
