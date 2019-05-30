"""
https://leetcode.com/problems/regular-expression-matching

Given an input string (s) and a pattern (p), implement regular expression matching with support for '.' and '*'.
"""


class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        """
        The idea is to treat this problem as a graph problem:
        - each position of the pattern is a node
        - there might be several nodes valid (with *) at s[i]

        When ingesting a new character, try to expand from every node:
        - as long as the set of valid node is not empty, continue
        - at the end, if the set of valid node contains the end node, match

        Rules for expanding from a node:
        - if character match with no stars: move one
        - if character match with stars: move one or stay
        - if star: you need to try following elements as well
        """

        nodes = []
        stars = []
        for c in p:
            if c == "*":
                stars[-1] = True
            else:
                nodes.append(c)
                stars.append(False)

        current_nodes = {0}
        for c in s:
            next_nodes = set()
            for n in current_nodes:
                while n < len(nodes):
                    if nodes[n] == c or nodes[n] == '.':
                        next_nodes.add(n + 1)
                        if stars[n]:
                            next_nodes.add(n)
                    if stars[n]:
                        n += 1
                    else:
                        break

            current_nodes = next_nodes
            if not current_nodes:
                return False

        # To take into account "a" matching "ab*"
        last_node = max(current_nodes)
        while last_node < len(nodes) and stars[last_node]:
            last_node += 1
        return len(nodes) == last_node
