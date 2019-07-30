"""
https://leetcode.com/problems/beautiful-arrangement/

Suppose you have N integers from 1 to N. We define a beautiful arrangement as an array that is constructed by these N
numbers successfully if one of the following is true for the ith position (1 <= i <= N) in this array:
* The number at the ith position is divisible by i.
* i is divisible by the number at the ith position.

Now given N, how many beautiful arrangements can you construct?
"""

class Solution:
    def countArrangement_1(self, n: int) -> int:
        """
        This looks like a back-tracking with pruning problem:
        - we have to try all combinations
        - we can abort prematuraly some prefix of solutions

        How de we know when to abort?
        - we just list the numbers that are valid at each position: initial phase in O(N ** 2)
        - we start with the indices with the least number of possible choices
        - we then remove the numbers from each places at each assignment at each iteration
        - we but branch everytime we see a set of possible go empty
        This takes 228 ms and beats 75%.
        """

        possibles = [set() for i in range(n)]
        for i in range(n):
            a = i + 1
            for j in range(n):
                b = j + 1
                if a % b == 0 or b % a == 0:
                    possibles[i].add(b)
        possibles.sort(key=lambda s: len(s))

        def try_remove_from(possible: int, start: int):
            indexes = []
            for i in range(start, n):
                if possible in possibles[i]:
                    if len(possibles) == 1:
                        return indexes, False
                    possibles[i].remove(possible)
                    indexes.append(i)
            return indexes, True

        def visit(pos: int) -> int:
            if pos == len(possibles):
                return 1
            count = 0
            for possible in possibles[pos]:
                indexes, valid = try_remove_from(possible, pos+1)
                if valid:
                    count += visit(pos + 1)
                for i in indexes:
                    possibles[i].add(possible)
            return count

        return visit(0)

    def countArrangement_2(self, n: int) -> int:
        """
        We can actually do better (and simpler): just avoid the number you already used.
        It cuts the branch slower, but it is faster to implement.
        It takes 136 ms and beats 95%.
        """

        possibles = [set() for i in range(n)]
        for i in range(n):
            a = i + 1
            for j in range(n):
                b = j + 1
                if a % b == 0 or b % a == 0:
                    possibles[i].add(b)
        possibles.sort(key=lambda s: len(s))

        def visit(taken, pos: int) -> int:
            if pos == len(possibles):
                return 1

            count = 0
            for possible in possibles[pos]:
                if possible not in taken:
                    taken.add(possible)
                    count += visit(taken, pos + 1)
                    taken.remove(possible)
            return count

        return visit(set(), 0)
