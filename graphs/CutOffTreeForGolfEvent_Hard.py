"""
https://leetcode.com/problems/cut-off-trees-for-golf-event/

You are asked to cut off trees in a forest for a golf event. The forest is represented as a non-negative 2D map, in this map:
* 0 represents the obstacle can't be reached.
* 1 represents the ground can be walked through.
* The place with number bigger than 1 represents a tree can be walked through, and this positive number represents the tree's height.

You are asked to cut off all the trees in this forest in the order of tree's height - always cut off the tree with lowest height first. And after cutting, the original place has the tree will become a grass (value 1).

You will start from the point (0, 0) and you should output the minimum steps you need to walk to cut off all the trees. If you can't cut off all the trees, output -1 in that situation.

You are guaranteed that no two trees have the same height and there is at least one tree needs to be cut off.
"""

from collections import deque
from typing import List


class Solution:
    def cutOffTree(self, forest: List[List[int]]) -> int:
        """
        Sort the points to get the list of coordinates to visit in order
        Then walk up the points in order and sum the distances
        (Precomputing the distance is of no use, since we never go back to a given point)
        """
        if not forest or not forest[0]:
            return -1

        if forest[0][0] == 0:
            return -1

        h = len(forest)
        w = len(forest[0])
        to_cut = [(0, 0)] + self.sort_by_height(forest, h, w)

        total_distance = 0
        for t1, t2 in zip(to_cut[:-1], to_cut[1:]):
            distance = self.distance(t1, t2, forest, h, w)
            if distance is None:
                return -1
            total_distance += distance
        return total_distance

    def sort_by_height(self, forest, h, w):
        to_cut = []
        for i in range(h):
            for j in range(w):
                if forest[i][j] > 1:
                    to_cut.append((i, j))
        to_cut.sort(key=lambda p: forest[p[0]][p[1]])
        return to_cut

    def distance(self, source, destination, forest, h, w):
        def neighbors(i, j):
            if i < h - 1:
                yield i + 1, j
            if i > 0:
                yield i - 1, j
            if j < w - 1:
                yield i, j + 1
            if j > 0:
                yield i, j - 1

        start_i, start_j = source
        discovered = {(start_i, start_j)}
        to_visit = deque([(start_i, start_j, 0)])
        while to_visit:
            i, j, dist = to_visit.popleft()
            if (i, j) == destination:
                return dist
            for x, y in neighbors(i, j):
                if (x, y) not in discovered and forest[x][y] > 0:
                    discovered.add((x, y))
                    to_visit.append((x, y, dist + 1))
        return None
