"""
https://leetcode.com/problems/maximum-width-of-binary-tree

Given a binary tree, write a function to get the maximum width of the given tree.
The width of a tree is the maximum width among all levels.
The binary tree has the same structure as a full binary tree, but some nodes are null.

The width of one level is defined as the length between the end-nodes
(the leftmost and right most non-null nodes in the level, where the null nodes between the end-nodes
are also counted into the length calculation.
"""


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def widthOfBinaryTree(self, root: TreeNode) -> int:
        """
        You cannot just subtract 1 or add 1 for each level: there is an exponential factor
        when descending down the tree.

        The idea is to use an indexing scheme for the position in the tree that follows the
        indexing scheme of a heap:
        - 2 * pos for left child
        - 2 * pos + 1 for right child

        Contrarily to the heap, you can start at position 0 for the root, because the indexes
        can overlap at each tree (in fact, you want to start at 0 to spare using too many IDs).

        Like this:

        0
        | \
        0   1
        |\  |\
        0 1 2 3

        Then do a simple DFS, and keep a list of the minimum indexes you found at each level.
        Compare with this minimum index at each node and keep track of the max width.

        Complexity is O(N) in time, and O(H) in space.
        """

        max_width = 0
        min_pos_by_level = []

        to_visit = [(root, 0, 0)]
        while to_visit:
            node, depth, position = to_visit.pop()
            if depth >= len(min_pos_by_level):
                min_pos_by_level.append(position)
                max_width = max(max_width, 1)
            else:
                max_width = max(max_width, position - min_pos_by_level[depth] + 1)
            if node.right:
                to_visit.append((node.right, depth + 1, 2 * position + 1))
            if node.left:
                to_visit.append((node.left, depth + 1, 2 * position))

        return max_width
