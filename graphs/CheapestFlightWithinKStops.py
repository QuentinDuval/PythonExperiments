"""
https://leetcode.com/problems/cheapest-flights-within-k-stops/

There are n cities connected by m flights. Each fight starts from city u and arrives at v with a price w.

Now given all the cities and flights, together with starting city src and the destination dst,
your task is to find the cheapest price from src to dst with up to k stops. If there is no such route, output -1.
"""


from collections import defaultdict
import heapq
from typing import List


class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        """
        Dijsktra will do, but we need to keep track of the number of stops as well
        in order to avoid to consider paths that are too long

        Beats 97%
        """

        graph = defaultdict(list)
        for u, v, w in flights:
            graph[u].append((v, w))

        distances = {src: 0}
        to_visit = [(0, 0, src)]  # price, nb_stops, node
        while to_visit:
            smallest_price, nb_stops, node = heapq.heappop(to_visit)
            if node == dst:
                return smallest_price

            distances[node] = smallest_price
            if nb_stops > k:
                continue

            for neighbor, cost in graph[node]:
                if neighbor not in distances:
                    heapq.heappush(to_visit, (smallest_price + cost, nb_stops + 1, neighbor))

        return -1
