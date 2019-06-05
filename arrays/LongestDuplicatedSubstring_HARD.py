class Solution:
    def longestDupSubstring_0(self, s: str) -> str:
        """
        Brute force is to:
        - try every single substring of size <= len(s) - 1 (they may overload)
        - put them into a set for each size, to try to find if already present

        Complexity is O(N ** 3):
        - N for each length
        - N for each starting point
        - N for the the hashing
        => Clearly, it will not fit.

        We could probably reuse the hash of some parts of the string (re-use of element of length - 1 to get O(N ** 2))
        """

        n = len(s)
        for l in reversed(range(1, n)):
            found = set()
            for i in range(n - l):  # i + l < n
                subs = s[i:i + l + 1]
                if subs in found:
                    return subs
                found.add(subs)
        return ""

    def longestDupSubstring_1(self, s: str) -> str:
        """
        SUFFIX TREES are awesome to find sub-strings: we can probably reuse the same technique here.

        We can try a suffix array, sort it, and then find the string the biggest prefix. But that would take O(N**2 log N)
        to create the array and sort, it, and O(N**2) to identify the bigger prefix

        Other solution is to create a TRIE, and then to collect the longest paths with number of leaf >= 2.

        We can do this during the creation of the TRIE:
        - Insert all strings s[i:] in the TRIE
        - Whenever you insert one, count the depth at which we went before forking
        - At the end of the process, you have the longest, and the prefix

        The complexity of creating a full TRIE is O(N ** 2).
        We can use Ukkonen algorithm to decrease this to O(N) and it would pass.
        """
        pass  # TODO

    def longestDupSubstring(self, s: str) -> str:
        """
        Use binary search to find the length of the longest substring:
        If a substring of size 1 <= X <= N if possible, it may be included in a bigger substring.

        If we did this, we still have to compute the hash in O(N) and try for each O(N) starts
        => O(N ** 2 * log N)

        But we can easily compute the hash in O(1) with a rolling hash.
        When there is a collision though, the comparison will take O(N).

        Time complexity becomes O(N log N) => TIMEOUT
        """
        from collections import defaultdict

        def find_duplicated(l: int) -> str:
            mod = 2 ** 64 - 1
            factor = 31
            factor_pow_l = 31 ** l

            rolling_hash = 0
            for i in range(l):
                rolling_hash = (rolling_hash * factor + ord(s[i])) % mod

            found = defaultdict(list)
            found[rolling_hash].append((0, l))

            for i in range(l, len(s)):
                rolling_hash = (rolling_hash * factor - ord(s[i - l]) * factor_pow_l + ord(s[i])) % mod
                start, end = i - l + 1, i + 1
                existing = found[rolling_hash]
                for b, e in existing:
                    if s[b:e] == s[start:end]:
                        return s[b:e]
                existing.append((start, end))
            return None

        lo = 1
        hi = len(s) - 1
        longest_duplicated = ""

        while lo <= hi:
            mid = lo + (hi - lo) // 2
            duplicated = find_duplicated(mid)
            if duplicated:
                lo = mid + 1
                longest_duplicated = duplicated
            else:
                hi = mid - 1

        return longest_duplicated
