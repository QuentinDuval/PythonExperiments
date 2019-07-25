"""
https://leetcode.com/problems/split-array-into-fibonacci-sequence/

Given a string S of digits, such as S = "123456579", we can split it into a Fibonacci-like sequence [123, 456, 579].

Formally, a Fibonacci-like sequence is a list F of non-negative integers such that:

* 0 <= F[i] <= 2^31 - 1, (that is, each integer fits a 32-bit signed integer type);
* F.length >= 3;
* F[i] + F[i+1] = F[i+2] for all 0 <= i < F.length - 2.

Also, note that when splitting the string into pieces, each piece must not have extra leading zeroes,
except if the piece is the number 0 itself.

Return any Fibonacci-like sequence split from S, or return [] if it cannot be done.
"""


from typing import List


class Solution:
    def splitIntoFibonacci(self, seq: str) -> List[int]:
        """
        Selecting the first two numbers is enough to then check if the sequence is valid
        => This means selecting two indices i and j (the index of the end of the two first integers)
        => We can try this systematically all i < j for a O(N ** 3) algorithm as the check is O(N)

        We can do even better by making sure not to test all i and j, but only those that would result in
        a number lower than 2 ^ 31 - 1.
        """

        max_len = len(str(2 ** 31 - 1))
        for i in range(1, min(len(seq), max_len + 1)):
            for j in range(i + 1, min(len(seq), i + max_len + 1)):
                if len(seq) - j < max(i, j - i):  # no room for another number
                    break

                a = int(seq[:i])
                b = int(seq[i:j])
                split = self.fib_split(a, b, seq[j:])
                if len(split) > 2:
                    return split
        return []

    def fib_split(self, a, b, seq):
        i = 0
        split = [a, b]
        while i < len(seq):
            a, b = b, a + b
            if b > 2 ** 31 - 1:
                return []

            prefix = str(b)
            for c in prefix:
                if i == len(seq) or seq[i] != c:
                    return []
                i += 1
            split.append(b)
        return split


