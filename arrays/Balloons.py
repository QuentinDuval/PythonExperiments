from typing import List


def findMinArrowShots(points: List[List[int]]) -> int:
    """
    https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/

    Important notes:

    Just merging the intervals and counting how many intervals there are does not work:
    - example of [1,4], [3, 6], [5, 8] => just one interval, but two arrows
    - this is actually a lower bound for the number of arrows

    Greedy strategy:
    Shoot the arrow that kills most of balloons
    - Just sort by the first coordinate, and then second coordinate
    - Scan from left to right
    - Eliminate all balloons whose start is below end of current balloon lot
      (The end of the balloon lot must be updated to the min of each balloon bursted)

    By the way, you can model the problem as a graph:
    - Each interval has an edge to intervals to which it is intersected
    - But it is not very useful...
    """
    points.sort()

    count = 0
    last = float('-inf')
    for p in points:
        if p[0] > last:
            last = p[1]
            count += 1
        elif p[1] < last:
            last = p[1]
    return count
