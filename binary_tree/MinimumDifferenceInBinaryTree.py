"""
https://leetcode.com/problems/minimum-absolute-difference-in-bst/

Given a binary search tree with non-negative values, find the minimum absolute difference between values of any two nodes.
"""

from binary_tree.TreeNode import *


class Solution:
    def getMinimumDifference(self, root: TreeNode) -> int:
        if not root:
            return 0

        def inorder(node: TreeNode):
            if node.left:
                yield from inorder(node.left)
            yield node.val
            if node.right:
                yield from inorder(node.right)

        def pairs(generator):
            prev = next(generator)
            for val in generator:
                yield prev, val
                prev = val

        return min(val - prev for prev, val in pairs(inorder(root)))
