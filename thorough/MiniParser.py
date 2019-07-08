"""
https://leetcode.com/problems/mini-parser/

Given a nested list of integers represented as a string, implement a parser to deserialize it.
Each element is either an integer, or a list -- whose elements may also be integers or other lists.

Note: You may assume that the string is well-formed:
* String is non-empty.
* String does not contain white spaces.
* String contains only digits 0-9, [, - ,, ].
"""


from typing import List, Tuple


# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
# class NestedInteger:
#    def __init__(self, value=None):
#        """
#        If value is not specified, initializes an empty list.
#        Otherwise initializes a single integer equal to value.
#        """
#
#    def isInteger(self):
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        :rtype bool
#        """
#
#    def add(self, elem):
#        """
#        Set this NestedInteger to hold a nested list and adds a nested integer elem to it.
#        :rtype void
#        """
#
#    def setInteger(self, value):
#        """
#        Set this NestedInteger to hold a single integer equal to value.
#        :rtype void
#        """
#
#    def getInteger(self):
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        :rtype int
#        """
#
#    def getList(self):
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        :rtype List[NestedInteger]
#        """

class Solution:
    def deserialize(self, s: str) -> NestedInteger:

        def parse_list(self, pos: int) -> Tuple[NestedInteger, int]:
            result = NestedInteger()
            end_pos = pos + 1
            if s[end_pos] == "]":
                return result, end_pos + 1

            while True:
                item, end_pos = parse(end_pos)
                result.add(item)
                if s[end_pos] == "]":
                    break
                end_pos += 1
            return result, end_pos + 1

        def parse_int(self, pos: int) -> Tuple[NestedInteger, int]:
            end_pos = pos + 1
            while end_pos < len(s) and s[end_pos].isdigit():
                end_pos += 1
            val = int(s[pos:end_pos])
            return NestedInteger(value=val), end_pos

        def parse(pos: int) -> Tuple[NestedInteger, int]:
            if s[pos].isdigit() or s[pos] == "-":
                return parse_int(s, pos)
            elif s[pos] == "[":
                return parse_list(s, pos)
            return None

        result, end_pos = parse(0)
        return result
