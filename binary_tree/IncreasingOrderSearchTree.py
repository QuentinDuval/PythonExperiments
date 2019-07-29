"""
https://leetcode.com/problems/increasing-order-search-tree/

Given a binary search tree, rearrange the tree in in-order so that the leftmost node in the tree is now the root of
the tree, and every node has no left child and only 1 right child.
"""

from binary_tree.TreeNode import *
from typing import Tuple


class Solution:
    def increasingBST(self, root: TreeNode) -> TreeNode:

        def visit(node: TreeNode) -> Tuple[TreeNode, TreeNode]:
            if node.left:
                first, last = visit(node.left)
                node.left = None
                last.right = node
                last = node
            else:
                first, last = node, node

            if node.right:
                first_right, last_right = visit(node.right)
                last.right = first_right
                last = last_right
            return first, last

        if root:
            return visit(root)[0]
        else:
            return None
