"""
https://leetcode.com/problems/count-complete-tree-nodes/

Given a complete binary tree, count the number of nodes.
"""


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def countNodes(self, root: TreeNode) -> int:
        """
        Recursively, check if the tree is complete on the left and on the right:
        - go all left (get the depth)
        - go all right (get the depth)

        If length left != length right on the left sub-tree, try right.
        Else, recurse on the left tree.

        Other technique:
        Use an indexing strategy for the tree, and binary search for the leaf!
        - get the last index = 2 * h - 1 by compute the height h
        - search the left using this index
        """
        if not root:
            return 0

        depth = self.get_max_depth(root)
        max_leaf_count = 2 ** (depth - 1)

        lo = 0
        hi = max_leaf_count - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            mid_depth = self.get_depth_of(root, depth, mid)
            if mid_depth == depth:
                lo = mid + 1
            else:
                hi = mid - 1
        # Max number of nodes - max leaf count + actual leaf count
        return 2 ** depth - 1 - max_leaf_count + lo

    def get_depth_of(self, root, depth, mid) -> int:
        if not root:
            return 0
        offset = 2 ** (depth - 2)
        if mid < offset:
            return 1 + self.get_depth_of(root.left, depth - 1, mid)
        else:
            return 1 + self.get_depth_of(root.right, depth - 1, mid - offset)

    def get_max_depth(self, root) -> int:
        depth = 0
        while root:
            depth += 1
            root = root.left
        return depth
