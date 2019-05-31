"""
https://leetcode.com/problems/minimum-number-of-refueling-stops/

A car travels from a starting position to a destination which is target miles east of the starting position.

Along the way, there are gas stations.  Each station[i] represents a gas station that is station[i][0] miles east of
the starting position, and has station[i][1] liters of gas.

The car starts with an infinite tank of gas, which initially has startFuel liters of fuel in it.
It uses 1 liter of gas per 1 mile that it drives.

When the car reaches a gas station, it may stop and refuel, transferring all the gas from the station into the car.

What is the least number of refueling stops the car must make in order to reach its destination?
If it cannot reach the destination, return -1.

Note that if the car reaches a gas station with 0 fuel left, the car can still refuel there.
If the car reaches the destination with 0 fuel left, it is still considered to have arrived.
"""


import heapq
from collections import deque
from typing import List


class Solution:
    def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
        """
        Solution 1:

        We can recast this to a BFS:
        - initial node is 0 with start_fuel, other nodes are the fuel stations
        - we can travel to other nodes if the fuel allows to reach them
        (the graph is not only the fuel stations, nodes also have the gaz tank content)

        Simply try all possibilities:
        - do not forget to add a 'discovered' state to avoid visiting multiple time the same node
        - stop as soon as 'target' is at reach
        - use binary search to easily collect all gaz stations that we can refuel in

        !!! CORRECT BUT TOO SLOW !!!
        """

        stations.append([0, startFuel])
        stations.sort()

        def b_search(lo: int, val: int):
            hi = len(stations) - 1
            while lo <= hi:
                mid = lo + (hi - lo) // 2
                if stations[mid][0] <= val:
                    lo = mid + 1    # will stop at stations[mid][0] > val
                else:
                    hi = mid - 1
            return lo

        to_visit = deque()
        to_visit.append((0, startFuel))
        discovered = set()

        def add(pos, fuel):
            if (pos, fuel) not in discovered:
                discovered.add((pos, fuel))
                to_visit.append((pos, fuel))

        refuel = 0
        while to_visit:
            for _ in range(len(to_visit)):
                pos, fuel = to_visit.popleft()
                if stations[pos][0] + fuel >= target:
                    return refuel

                hi = b_search(pos, stations[pos][0] + fuel)
                for i in range(pos + 1, hi):
                    distance = stations[i][0] - stations[pos][0]
                    add(i, fuel - distance + stations[i][1])
            refuel += 1
        return -1

class Solution:
    def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
        """
        Solution 2:

        Recast the problem to something else: starting from position 0, we want the minimum of refill to have had
        enough 'total_fuel' to reach the target.

        Simply proceed recursively:
        - Given the current amount of fuel, look all the stations you can reach
        - Select the one with the highest amount of fuel, refill there
        - Look at all the stations you can reach now (minus the one you used) and repeat until enough fuel

        To avoid visiting the same stations twice, and to easily retrieve the max refill, use a HEAP

        Beats 97% (44 ms)
        """
        stations.append([0, startFuel])
        stations.sort()

        def b_search(lo: int, val: int):
            hi = len(stations) - 1
            while lo <= hi:
                mid = lo + (hi - lo) // 2
                if stations[mid][0] <= val:
                    lo = mid + 1  # will stop at stations[mid][0] > val
                else:
                    hi = mid - 1
            return lo

        fuel_heap = []              # List of refills available
        furthest = 0                # To avoid adding the same stations twice in the HEAP
        total_fuel = startFuel
        refill_nb = 0

        while True:
            if total_fuel >= target:
                return refill_nb

            # Add the new stations that are reachable with our total_fuel (furthest tacks the already added ones)
            hi = b_search(furthest, total_fuel)
            for i in range(furthest + 1, hi):
                heapq.heappush(fuel_heap, -1 * stations[i][1])

            if not fuel_heap:
                return -1

            # Select the highest refill (beware, heapq is MIN heap)
            max_possible_refill = -1 * heapq.heappop(fuel_heap)
            total_fuel += max_possible_refill
            furthest = hi - 1
            refill_nb += 1

        return -1
