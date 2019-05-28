"""
https://leetcode.com/problems/binary-tree-cameras

Given a binary tree, we install cameras on the nodes of the tree.
Each camera at a node can monitor its parent, itself, and its immediate children.
Calculate the minimum number of cameras needed to monitor all nodes of the tree.
"""


from enum import Enum


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Status(Enum):
    CAMERA = 0
    COVERED = 1
    NOT_COVERED = 2


class Solution:
    def minCameraCover(self, root: TreeNode) -> int:
        """
        The key is to realise that:
        1. the leaves must be covered
        2. coloring a leaf is always sub-optimal
        (The best way to see this is to DRAW THE PICTURE...)

        So the key is to recurse, and look at the sub-solutions:
        - If either child is not covered, you need to put a camera
        - Otherwise, if a child has a camera, you can get out without a camera
        - Otherwise, return not covered (case of the leaves)

        Time complexity is O(N) for we have to analyse all nodes
        Space complexity is O(H) which H being the height of the tree
        """

        if not root:
            return 0
        count, status = self.visit(root)
        if status == Status.NOT_COVERED:
            count += 1
        return count

    def visit(self, node: TreeNode):
        count = 0
        status = Status.NOT_COVERED

        for child in [node.left, node.right]:
            if child is not None:
                sub_count, sub_status = self.visit(child)
                count += sub_count
                if sub_status == Status.NOT_COVERED:
                    status = Status.CAMERA
                elif sub_status == Status.CAMERA and status != Status.CAMERA:
                    status = Status.COVERED

        if status == Status.CAMERA:
            count += 1
        return count, status
