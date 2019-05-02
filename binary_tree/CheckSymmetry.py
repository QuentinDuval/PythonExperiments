"""
https://leetcode.com/problems/symmetric-tree/

Note that doing a in-order, post-order, or pre-order traversal would not work
- There are several trees that can lead to the same order
- We could obviously combine several traversal, but that would not be efficient

So the best thing is to visit the tree and check one by one
"""


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def isSymmetric(self, root):
        def check(l, r):
            if l is None: return r is None
            if r is None: return l is None
            return l.val == r.val and check(l.right, r.left) and check(l.left, r.right)

        if root is None: return True
        return check(root.left, root.right)
