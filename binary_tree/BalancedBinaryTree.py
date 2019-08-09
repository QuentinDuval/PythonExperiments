from typing import Tuple


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        if not root:
            return True

        def visit(node: TreeNode) -> Tuple[bool, int]:
            l_depth = r_depth = 0
            if node.left:
                balanced, l_depth = visit(node.left)
                if not balanced:
                    return False, 0
            if node.right:
                balanced, r_depth = visit(node.right)
                if not balanced:
                    return False, 0
            return abs(l_depth - r_depth) <= 1, 1 + max(l_depth, r_depth)

        return visit(root)[0]
