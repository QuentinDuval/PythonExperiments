"""
https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/

Given the root of a binary tree, find the maximum value V for which there exists different nodes A and B
where V = |A.val - B.val| and A is an ancestor of B.

(A node A is an ancestor of B if either: any child of A is equal to B, or any child of A is an ancestor of B.)
"""


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def maxAncestorDiff(self, root: TreeNode) -> int:
        """
        When going down the descendants:
        - keep the minimum value of ancestor
        - keep the maximum value of ancestor
        => Allows to do the diff

        When going up (end of recursion):
        - look at the max difference on left and right
        - take the max with own difference with above

        Complexity is O(N)
        """

        if not root:
            return 0

        def recur(node: TreeNode, min_ancestor: int, max_ancestor: int) -> int:
            max_diff = max(abs(node.val - min_ancestor), abs(node.val - max_ancestor))
            min_ancestor = min(min_ancestor, node.val)
            max_ancestor = max(max_ancestor, node.val)
            if node.left:
                max_diff = max(max_diff, recur(node.left, min_ancestor, max_ancestor))
            if node.right:
                max_diff = max(max_diff, recur(node.right, min_ancestor, max_ancestor))
            return max_diff

        return recur(root, root.val, root.val)

    def maxAncestorDiff(self, root: TreeNode) -> int:
        """
        Without using recursion (faster)
        """

        if not root:
            return 0

        max_diff = 0
        to_visit = [(root, root.val, root.val)]
        while to_visit:
            node, min_ancestor, max_ancestor = to_visit.pop()
            max_diff = max(max_diff,                            # Do not forget the max_diff here
                           abs(node.val - min_ancestor),
                           abs(node.val - max_ancestor))
            min_ancestor = min(min_ancestor, node.val)
            max_ancestor = max(max_ancestor, node.val)
            if node.right:
                to_visit.append((node.right, min_ancestor, max_ancestor))
            if node.left:
                to_visit.append((node.left, min_ancestor, max_ancestor))
        return max_diff
