from collections import defaultdict
from typing import List


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def findFrequentTreeSum(self, root: TreeNode) -> List[int]:
        """
        The trick is to compute efficiently the sub-sums
        We can do it in just one visit
        """
        if root is None:
            return []

        frequencies = defaultdict(int)

        def visit(node: TreeNode) -> int:
            if node is None:
                return 0

            total = node.val
            if node.left:
                total += visit(node.left)
            if node.right:
                total += visit(node.right)
            frequencies[total] += 1
            return total

        visit(root)
        max_freq = max(frequencies.values())
        return [val for val, count in frequencies.items() if count == max_freq]
