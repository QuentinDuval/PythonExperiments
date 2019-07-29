"""
https://practice.geeksforgeeks.org/problems/minimum-platforms/0

Given arrival and departure times of all trains that reach a railway station.
Your task is to find the minimum number of platforms required for the railway station so that no train waits.
"""


from typing import *


def min_platforms(arrivals: List[Tuple[int, int]], departures: List[Tuple[int, int]]) -> int:
    """
    The goal is to find how many intersecting intervals there are at worse.

    One idea:
    - collect all interesting point (arrival and departure) in order
    - add +1 at arrival
    - remove -1 at departure
    - count the maximum you reach

    Beware when falls on same hour: take into account the arrival first!
    """

    '''
    # Too slow (although it is O(N log N))
    changes = [(time, +1) for time in arrivals]
    changes.extend((time, -1) for time in departures)
    changes.sort(key=lambda p:(p[0], -p[1]))

    count = 0
    max_count = 0
    for time, diff in changes:
        count += diff
        max_count = max(max_count, count)
    return max_count
    '''

    """
    We can use a modified merge (of merge sort) to go faster and avoid
    having to see all the 'departures' array
    """
    count = 0
    max_count = 0

    arrivals.sort()
    departures.sort()

    arrival = 0
    departure = 0
    while arrival < len(arrivals):
        if arrivals[arrival] <= departures[departure]:
            count += 1
            arrival += 1
            max_count = max(count, max_count)
        else:
            count -= 1
            departure += 1

    # We do not care about remaining departures, they only decrease the max
    return max_count



def read_times(s: str, n: int):
    times = []
    for token in s.split(" "):
        if token:
            h = int(token[:2])
            m = int(token[2:])
            times.append((h, m))
    return times


nb_tests = int(input())
for _ in range(nb_tests):
    n = int(input())
    arrivals = read_times(input(), n)
    departures = read_times(input(), n)
    print(min_platforms(arrivals, departures))
