"""
https://leetcode.com/problems/maximum-length-of-repeated-subarray/

Given two integer arrays A and B, return the maximum length of an subarray that appears in both arrays.
"""

from typing import List


class Solution:
    def findLength_dp(self, a: List[int], b: List[int]) -> int:
        """
        Brute force solution is to try every starting point on A and B and try to match:
        - O(N) starting point on A
        - O(M) starting point on B
        - O(N) check cost
        => O(N ** 2 * M)
        (Note: we could do even worse: selecting a substring in A and try to search it in B)

        We could go for dynamic programming with O(N * M) starting points, but we have to be careful:
        we are not looking for the longest sub-sequence, but the longest sub-array => continuity.

        So the recursion is different and to be tried for every i, j:

            max_len(i, j) =
                if i == len(a) or j == len(b):
                    return 0

                if a[i] == b[j]:
                    return 1 + maximize_from(i+1, j+1)
                return 0

        Number of sub-problems: O(N*M)
        - Time complexity: O(N*M)
        - Space complexity: O(N*M) for top-bottom or O(min(N, M)) for bottom-up

        Beats 70% (93% for memory)
        """

        '''
        @lru_cache(maxsize=None)
        def maximize_from(i: int, j: int):
            if i == len(a) or j == len(b):
                return 0

            if a[i] == b[j]:
                return 1 + maximize_from(i+1, j+1)
            return 0

        max_len = 0
        for i in reversed(range(len(a))):
            for j in reversed(range(len(b))):
                max_len = max(max_len, maximize_from(i, j))
        return max_len
        '''

        '''
        # O(N * M) memory usage
        max_len = 0
        memo = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in reversed(range(len(a))):
            for j in reversed(range(len(b))):
                if a[i] == b[j]:
                    memo[i][j] = 1 + memo[i+1][j+1]
                max_len = max(max_len, memo[i][j])
        return max_len
        '''

        # O(min(N, M)) memory usage
        if len(a) < len(b):
            a, b = b, a

        max_len = 0
        memo = [0] * (len(b) + 1)
        for i in reversed(range(len(a))):
            new_memo = [0] * (len(b) + 1)
            for j in reversed(range(len(b))):
                if a[i] == b[j]:
                    new_memo[j] = 1 + memo[j + 1]
                    max_len = max(max_len, new_memo[j])
            memo = new_memo
        return max_len

    def findLength_kmp(self, a: List[int], b: List[int]) -> int:
        """
        For every beginning of 'a', do a KMP search on the array 'b'.
        Keep track of the maximum index you found in the KMP state machine.

        => Time complexity is O(M * (M + N))
        => Space complexity is O(M)
        """
        # TODO

    def findLength_hm(self, a: List[int], b: List[int]) -> int:
        """
        Solution based on hash maps:
        - create all sub-array of a, compute their hash, and put this in a set 'seen'
        - try all sub-array of b and check if it belongs in the set 'seen'

        You have to be careful with this in how you compute the hashes:
        - the brute force would take O(N) leading to a O(N**3) algorithm
        - with rolling hash, the complexity is O(min(N**2, M**2))
        """
        pass  # TODO

    def findLength(self, a: List[int], b: List[int]) -> int:
        """
        Binary search on the length of the longest sub-string:
        - If we have length L1 and not length L2, try in-between to find the highest length
        - Start between length 0 and O(min(N, M))

        Brute force application of the idea is O(N * M * log(min(N, M)))

        The idea is to use a hash-map to do better:
        - compute a rolling hash of all windows of size L in a
        - search for window of size L in this set (with rolling hashes)

        Performance should be O(max(N, M) * log(min(N, M)))
        """
        pass  # TODO

    def findLength_trie(self, a: List[int], b: List[int]) -> int:
        """
        Play with generalized suffix trees (trie)
        Performance should be O(N)
        """
        pass  # TODO
