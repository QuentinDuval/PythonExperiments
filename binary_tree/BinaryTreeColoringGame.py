"""
https://leetcode.com/problems/binary-tree-coloring-game/

Two players play a turn based game on a binary tree.
We are given the root of this binary tree, and the number of nodes n in the tree.

n is odd, and each node has a distinct value from 1 to n.

Initially, the first player names a value x with 1 <= x <= n, and the second player names a value y with 1 <= y <= n
and y != x.  The first player colors the node with value x red, and the second player colors the node with value y blue.

Then, the players take turns starting with the first player.  In each turn, that player chooses a node of their color
(red if player 1, blue if player 2) and colors an uncolored neighbor of the chosen node
(either the left child, right child, or parent of the chosen node.)

If (and only if) a player cannot choose such a node in this way, they must pass their turn.
If both players pass their turn, the game ends, and the winner is the player that colored more nodes.

You are the second player.  If it is possible to choose such a y to ensure you win the game, return true.
If it is not possible, return false.
"""


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def btreeGameWinningMove(self, root: TreeNode, n: int, x: int) -> bool:
        """
        x places its moves at one node, then we should block as much as x
        movement as possible, by selecting either its parent, left or right child

        x will win the 2 parts that we did not selected (it cuts the way)
        => y can only win if one part is stictly bigger than the other combined + 1

        count the total number of nodes T
        count the left tree node count L
        count the right tree node count R
        parent node count P = T - 1 - L - R

        y can only win if either P > T / 2 or L > N / 2 or R > N / 2
        """
        if not root:
            return False

        counts = [0] * 3
        to_visit = [(root, 0)]
        while to_visit:
            node, part = to_visit.pop()
            counts[part] += 1
            found = node.val == x
            if node.left:
                to_visit.append((node.left, 1 if found else part))
            if node.right:
                to_visit.append((node.right, 2 if found else part))
        counts[0] -= 1  # for the x first move
        return any(count > n / 2 for count in counts)

