"""
https://leetcode.com/problems/camelcase-matching/

A query word matches a given pattern if we can insert lowercase letters to the pattern word so that it equals the query.
(We may insert each character at any position, and may insert 0 characters.)

Given a list of queries, and a pattern, return an answer list of booleans, where answer[i] is true if and only if
queries[i] matches the pattern.
"""


from typing import List


class Solution:
    def camelMatch(self, queries: List[str], pattern: str) -> List[bool]:

        def match(query):
            pi = 0
            for c in query:
                if pi == len(pattern):
                    if not c.islower():
                        return False
                elif pattern[pi] == c:
                    pi += 1
                elif not c.islower():
                    return False
            return pi == len(pattern)

        return [match(query) for query in queries]
