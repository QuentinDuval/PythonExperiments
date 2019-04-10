"""
https://leetcode.com/problems/find-bottom-left-tree-value

Given a binary tree, find the leftmost value in the last row of the tree.

"""


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def findBottomLeftValue(self, root: TreeNode) -> int:
        """
        - Search left: find the deepest node
        - Search right: find the deepest node
        - Keep only the nodes that are strictly deeper
        """

        def visit(depth: int, node: TreeNode):
            l_depth, l_val = visit(depth + 1, node.left) if node.left else (depth, node.val)
            r_depth, r_val = visit(depth + 1, node.right) if node.right else (depth, node.val)
            if l_depth < r_depth:
                return r_depth, r_val
            else:
                return l_depth, l_val

        _, val = visit(0, root)
        return val
