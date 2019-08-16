"""
https://leetcode.com/problems/check-completeness-of-a-binary-tree/

Given a binary tree, determine if it is a complete binary tree.

Definition of a complete binary tree from Wikipedia:

In a complete binary tree every level, except possibly the last, is completely filled,
and all nodes in the last level are as far left as possible.
It can have between 1 and 2h nodes inclusive at the last level h.
"""


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


from typing import Tuple


class Solution:
    def isCompleteTree(self, root: TreeNode) -> bool:
        """
        Follow a number scheme in the Tree such that:
        - root is tagged 1
        - left child is tagged 2 * father id
        - right child is tagged 2 * father id + 1

        Then just count the nodes of the tree, and check that
        the last ID matches the count in the tree.
        """

        def visit(node: TreeNode, pos: int) -> Tuple[int, int]:
            if not node:
                return 0, 0

            l_id, l_count = visit(node.left, 2 * pos)
            r_id, r_count = visit(node.right, 2 * pos + 1)
            return max(pos, l_id, r_id), 1 + l_count + r_count

        last_id, count = visit(root, 1)
        return last_id == count
