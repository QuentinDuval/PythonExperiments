"""
https://leetcode.com/problems/distribute-coins-in-binary-tree/

Given the root of a binary tree with N nodes, each node in the tree has node.val coins, and there are N coins total.

In one move, we may choose two adjacent nodes and move one coin from one node to another.
(The move may be from parent to child, or from child to parent.)

Return the number of moves required to make every node have exactly one coin.
"""


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    # TODO - to actually be tested

    def distributeCoins(self, root: TreeNode) -> int:
        """
        The key here is to draw the tree, and see that the problem can be solved recursively

        At node N, if left tree has excedent X, and right tree has excedent Y (both X and Y can be negative),
        then the node N, has excedent X + Y and the flow from or to this node is abs(X + Y) moves.
        """

        moves = 0

        def visit(node: TreeNode) -> int:
            nonlocal moves
            if not node:
                return 0

            left = visit(node.left)
            right = visit(node.right)
            moves += abs(left + right)
            return left + right

        return moves
