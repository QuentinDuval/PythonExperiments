"""
https://leetcode.com/problems/shortest-palindrome

Given a string s, you are allowed to convert it to a palindrome by adding characters in front of it.
Find and return the shortest palindrome you can find by performing this transformation.
"""


class Solution:
    def shortestPalindrome(self, s: str) -> str:
        """
        You can only add characters in front of it:
        - it is a matter of selecting the center, to replicate the right part to the left
        - the center has to be in the left part of s

        It ressemble a sub-string search problem, in which you
        would search the reversed 's' in 's' and see how far you can go

        At this point, you have a kind of back-tracking of left pointer:

        aacecaaa
          *  *

        With this we can find the padding to add left.
        From this padding, we can return the new string.

        Brute force will give us O(N ** 2) and will timeout - BELOW
        """

        for pad_left in range(len(s)):
            lo = 0
            hi = len(s) - pad_left - 1
            while lo <= hi and s[lo] == s[hi]:
                lo += 1
                hi -= 1
            if lo > hi:
                return s[::-1][:pad_left] + s

        return ""

    def shortestPalindrome(self, s: str) -> str:
        """
        With the right state machine (KMP), we should be able to do it in O(N)
        Reverse the string a pre-process it to deduce the state machine?

        If we look for the reversed string at the end of the string, we could use this:
        - this would give us a number of charater to skip at the end
        - BUT we have to stop the string match at the right size (if reach end of string)

          a a c e c a a a .
        a 1 2 2 1 1 6 7 8 -
        c 0 0 3 0 5 0 0 3
        e 0 0 0 4 0 0 0 0
                ^
        we matched "aac" and receive a 'a', 'c' or 'e'

        How to construct the KMP state machine?
        - We estimate where we would be had we had started one character after
        - We keep a index to the state we had, had we started one character after
        - We use this index to consult where to go next in case of error
          (just copy the row of prev state - and replace for the real transition)
        """

        if not s:
            return ""

        # Compute the KMP state machine

        char_set = list(set(s))
        char_pos = {char_set[i]: i for i in range(len(char_set))}

        kmp = [[0] * len(char_set) for _ in range(len(s))]
        kmp[0][char_pos[s[0]]] = 1

        back_up_state = 0  # where we would be, had we started one character after
        for state in range(1, len(s)):
            kmp[state] = kmp[back_up_state][:]
            kmp[state][char_pos[s[state]]] = state + 1
            back_up_state = kmp[back_up_state][char_pos[s[state]]]

        """
        Search the position of 's' in the reverse of 's'

        Example:
        - search position of 'aacecaaa' in 'aaacecaa'
        - we would like to find 1, and this gives us the shift
        But the length of the pattern makes it impossible!

        Solution 1:
        Multiply the reversed string by 2.
        Example:
        - search position of 'aacecaaa' in 'aaacecaaaaacecaa'
        - we will find 1 :)
        Example:
        - search position of 'abcd' in 'bcdabcda'
        - we find nothing :(

        Solution 2:
        Stop when we match the end of the string.
        """

        pad_left = None

        state = 0
        rev_s = s[::-1]
        for i, c in enumerate(rev_s):
            state = kmp[state][char_pos[c]]
            if state == len(kmp):  # match !
                pad_left = i + 1 - len(kmp)  # where we first stated to match
                break

        if pad_left is None:
            pad_left = len(rev_s) - state
        return s[::-1][:pad_left] + s
