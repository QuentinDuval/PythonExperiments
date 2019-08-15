"""
https://leetcode.com/problems/corporate-flight-bookings/

There are n flights, and they are labeled from 1 to n.

We have a list of flight bookings.

The i-th booking bookings[i] = [i, j, k] means that we booked k seats from flights labeled i to j inclusive.

Return an array answer of length n, representing the number of seats booked on each flight in order of their label.
"""


from typing import List


class Solution:
    def corpFlightBookings_bit(self, bookings: List[List[int]], n: int) -> List[int]:
        """
        The only way to do something reasonnably efficient enough here is to
        avoid searching the bookings for each N.

        We could have a kind of tree to summarize the bookings.
        => You can go for a BIT (Binary Index Tree).

        Normally a BIT works as follows:
        - query for a range by removing the last bit until reaching zero
        - add a number by adding the last bit (contibute to all including ranges)

        Here the BIT will work the other way around:
        - query for a position by adding the last bit (sum all including ranges)
        - add a number to a range by removing the last bit until reaching zero

        The BIT normally works by interval [0:x], so to add an interval [a, b]:
        - add to [0:b]
        - remove to [0:a-1]

        Complexity is O(N log N) in time and O(N) in space.
        => Beat only 5% (1500 ms)

        Note: We could compress the data (intersection of bookings) but complex.
        """

        bit = [0] * (n + 1)

        def add_up_to(end: int, val: int):
            while end > 0:
                bit[end] += val
                end -= (end & -end)  # Removes the last bit

        def query_at(pos: int) -> int:
            total = 0
            while pos < len(bit):
                total += bit[pos]
                pos += (pos & -pos)  # Add the last bit
            return total

        for start, end, count in bookings:
            add_up_to(end, count)
            add_up_to(start - 1, -count)
        return [query_at(i) for i in range(1, n + 1)]

    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        """
        In fact, we can summarize the bookings much more efficiently by just using
        a prefix sum based idea.

        Again, just like the BIT, the idea is a bit twisted.
        - when we add interval [a, b] value K, we put +K at place a, and -K at place b+1
        - then we just sum the value at each position to answer the question

        Complexity is O(N)
        => Beat only 16% (1056 ms)
        """

        '''
        diffs = [0] * (n+2)
        for start, end, count in bookings:
            diffs[start] += count
            diffs[end+1] -= count

        res = []
        prefix_sum = 0
        for i in range(1, n+1):
            prefix_sum += diffs[i]
            res.append(prefix_sum)
        return res
        '''

        """
        Slight improvement in terms of optimization
        => Beats 70% (988 ms)
        """

        # TODO - it is interesting to see how the pattern of query influences the complexity (you cannot beat the BIT but if you take in order)

        diffs = [0] * (n + 1)
        for start, end, count in bookings:
            diffs[start - 1] += count
            diffs[end] -= count

        for i in range(1, n):
            diffs[i] += diffs[i - 1]
        return diffs[:-1]


