"""
https://leetcode.com/problems/find-largest-value-in-each-tree-row/

You need to find the largest value in each row of a binary tree.
"""

from binary_tree.TreeNode import *
from typing import *


class Solution:
    def largestValues(self, root: TreeNode) -> List[int]:
        """
        There are several ways to do it:
        - DFS and keep a list of the number already found at that path
        - BFS and just do a max on each row you find
        """

        if not root:
            return []

        '''
        # BFS here (64 ms - beats 20%)
        result = []
        stage = [root]
        while stage:
            next_stage = []
            maximum = stage[0].val
            for node in stage:
                maximum = max(maximum, node.val)
                if node.left:
                    next_stage.append(node.left)
                if node.right:
                    next_stage.append(node.right)
            stage = next_stage
            result.append(maximum)
        return result
        '''

        # DFS - 52 ms (beats 87%)
        result = []

        def dfs(node: TreeNode, depth: int):
            if depth >= len(result):
                result.append(node.val)
            else:
                result[depth] = max(result[depth], node.val)

            if node.left:
                dfs(node.left, depth + 1)
            if node.right:
                dfs(node.right, depth + 1)

        dfs(root, 0)
        return result
