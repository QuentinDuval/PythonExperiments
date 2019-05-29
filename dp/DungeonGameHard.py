"""
https://leetcode.com/problems/dungeon-game
"""


# TODO - does not work - finish it


def trace(f):
    def wrapped(*args):
        res = f(*args)
        print(f, ":", args, "=>", res)
        return res

    return wrapped


class MaxHeap:
    # import heapq
    # Because 'heapq' return the smallest
    # Because we want a replacement strategy

    def __init__(self):
        self.position = {}
        self.heap = [(float('inf'), None)]  # i must be > 2 * i and 2 * i + 1

    def __len__(self):
        return len(self.position)

    @trace
    def push(self, prio, val):
        pos = self.position.get(val)
        if pos is not None:
            prev_prio, val = self.heap[pos]
            self.heap[pos] = (prio, val)
            if prev_prio < prio:
                self.swim_up(pos)
            else:
                self.sink_down(pos)
        else:
            pos = len(self.heap)
            self.position[val] = pos
            self.heap.append((prio, val))
            self.swim_up(pos)

    @trace
    def pop(self):
        self.swap(1, len(self.heap) - 1)
        prio, val = self.heap.pop()
        del self.position[val]
        self.sink_down(1)
        return prio, val

    def swim_up(self, pos):
        father = pos // 2
        if self.heap[father][0] < self.heap[pos][0]:
            self.swap(father, pos)
            self.swim_up(father)

    def sink_down(self, pos):
        higher_child = None
        if 2 * pos < len(self.heap):
            higher_child = 2 * pos
        if 2 * pos + 1 < len(self.heap):
            if self.heap[2 * pos + 1][0] > self.heap[higher_child][0]:
                higher_child = 2 * pos + 1

        if higher_child and self.heap[higher_child][0] > self.heap[pos][0]:
            self.swap(higher_child, pos)
            self.sink_down(higher_child)

    def swap(self, pos1, pos2):
        self.heap[pos1], self.heap[pos2] = self.heap[pos2], self.heap[pos1]
        for pos in [pos1, pos2]:
            self.position[self.heap[pos][1]] = self.heap[pos][0]


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
        """

        if not grid or not grid[0]:
            return 1

        h = len(grid)
        w = len(grid[0])
        princess = (h - 1, w - 1)

        def neighbors(x, y):
            if x < h - 1:
                yield x + 1, y
            if y < w - 1:
                yield x, y + 1

        min_hp = 0
        to_visit = MaxHeap()
        to_visit.push(grid[0][0], (0, 0))

        while to_visit:
            hp, (x, y) = to_visit.pop()
            min_hp = min(min_hp, hp)
            if (x, y) == princess:
                break

            for i, j in neighbors(x, y):
                to_visit.push(hp + grid[i][j], (i, j))

        return 1 - min_hp
