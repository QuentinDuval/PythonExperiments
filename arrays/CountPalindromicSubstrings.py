"""
https://leetcode.com/problems/palindromic-substrings

Given a string, your task is to count how many palindromic substrings in this string.

The substrings with different start indexes or end indexes are counted as different substrings even they consist of same characters.
"""



class Solution:
    def countSubstrings(self, s: str) -> int:
        """
        The naive algorithm consists in testing all pairs i < j and check the reversal
        Complexity: O(N**3) - It passes!
        """

        '''
        count = 0
        n = len(s)
        for i in range(n):
            for j in range(i+1, n+1):
                if s[i:j] == s[i:j][::-1]:
                    count += 1
        return count
        '''

        """
        Observations:
        - if s[i:j] is a palindrome, so it s[i+1:j-1]
        - there are N possible center of the palindrom
        => So try different centers!
        
        Complexity: O(N**2)
        """

        count = 0
        n = len(s)
        for mid in range(n):
            for hi in [mid, mid + 1]:
                lo = mid
                while lo >= 0 and hi < n and s[lo] == s[hi]:
                    count += 1
                    lo -= 1
                    hi += 1
        return count
