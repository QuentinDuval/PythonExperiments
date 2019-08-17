"""
https://leetcode.com/problems/verify-preorder-serialization-of-a-binary-tree/

One way to serialize a binary tree is to use pre-order traversal. When we encounter a non-null node, we record
the node's value. If it is a null node, we record using a sentinel value such as #.

     _9_
    /   \
   3     2
  / \   / \
 4   1  #  6
/ \ / \   / \
# # # #   # #

For example, the above binary tree can be serialized to the string "9,3,4,#,#,1,#,#,2,#,6,#,#",
where # represents a null node.

Given a string of comma separated values, verify whether it is a correct preorder traversal serialization of a binary tree.
Find an algorithm without reconstructing the tree.

Each comma separated value in the string must be either an integer or a character '#' representing null pointer.

You may assume that the input format is always valid.
"""

from typing import Tuple


class Solution:
    def isValidSerialization(self, s: str) -> bool:
        """
        As for all try problems, a recursive solution is always to be tried.
        """
        s = s.split(',')

        def verify(pos: int) -> Tuple[bool, int]:
            if pos == len(s):
                return False, pos
            if s[pos] == '#':
                return True, pos + 1
            valid, pos = verify(pos + 1)
            if not valid:
                return valid, pos
            return verify(pos)

        valid, pos = verify(0)
        return valid and pos == len(s)
