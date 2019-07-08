"""
https://leetcode.com/problems/smallest-string-starting-from-leaf/

Given the root of a binary tree, each node has a value from 0 to 25 representing the letters 'a' to 'z'.
Find the lexicographically smallest string that starts at a leaf of this tree and ends at the root.
"""


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def smallestFromLeaf(self, root: TreeNode) -> str:
        if not root:
            return ""

        def to_letter(n: int) -> str:
            return chr(ord('a') + n)

        def is_leaf(node: TreeNode):
            return not node.left and not node.right

        def visit(node: TreeNode, current: str):
            current += to_letter(node.val)  # String are immutable, otherwise you would have to think about POP()
            if is_leaf(node):
                return current[::-1]
            else:
                smallest = None
                if node.left:
                    smallest = visit(node.left, current)
                if node.right:
                    rhs = visit(node.right, current)
                    smallest = min(smallest, rhs) if smallest else rhs
                return smallest

        return visit(root, "")
