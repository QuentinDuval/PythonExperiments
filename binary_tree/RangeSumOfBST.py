"""
https://leetcode.com/problems/range-sum-of-bst

Given the root node of a binary search tree, return the sum of values of all nodes with value between L and R (inclusive).

The binary search tree is guaranteed to have unique values.
"""


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def rangeSumBST(self, root: TreeNode, lo: int, hi: int) -> int:
        """
        One solution would be to do an in-order traversal and sum the values along the way (or just recursive sum along the tree).
        => O(N) but in case the range [lo,hi] is small, this is wasteful.

        Another solution is to look for 'lo' in the binary tree:
        - then do an inorder traversal from there
        - and stop when we meet 'hi'
        => This is a bit complex

        A simpler solution is to recursively compute the sum of sub-trees, using lo and hi to direct the visit.
        """

        def visit(node: TreeNode) -> int:
            if not node:
                return 0
            if node.val < lo:
                return visit(node.right)
            elif hi < node.val:
                return visit(node.left)
            else:
                return node.val + visit(node.left) + visit(node.right)
        return visit(root)
