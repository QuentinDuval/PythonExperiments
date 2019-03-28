from typing import List

"""
https://leetcode.com/problems/brick-wall/

There is a brick wall in front of you. The wall is rectangular and has several rows of bricks.
The bricks have the same height but different width.
You want to draw a vertical line from the top to the bottom and cross the least bricks.

The brick wall is represented by a list of rows.
Each row is a list of integers representing the width of each brick in this row from left to right.

If your line go through the edge of a brick, then the brick is not considered as crossed.
You need to find out how to draw the line to cross the least bricks and return the number of crossed bricks.
"""


class Solution:
    def leastBricks(self, wall: List[List[int]]) -> int:
        """
        Scan the lists one-by-one
        - add +1 to each brick in an interval
        - select the one with the smallest value

        Complexity: O(H * W) with
        - H < 1e4
        - W < 4e9
        So clearly this is too slow

        ------------------------------------------------------------------

        Instead, you should REVERSE the logic.

        You only consider places in which there is at least one block stop:
        - add +1 to each interval you found
        - select the one with the highest score

        This way, the number of intervals is at maximum the number of bricks
        Which is O(B * H) with:
        - H < 1e4
        - W < 1e4

        ------------------------------------------------------------------

        Can we do better? Not really, we have to analyze each brick.
        """
        if not wall or not wall[0]:
            return 0

        holes = {}
        wall_height = len(wall)

        for row in wall:
            position = 0
            for width in row[:-1]:
                position += width
                holes[position] = holes.get(position, 0) + 1

        min_crossed = wall_height
        for skipped in holes.values():
            min_crossed = min(min_crossed, wall_height - skipped)
        return min_crossed
