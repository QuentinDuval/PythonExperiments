"""
https://leetcode.com/problems/path-sum-iii/

You are given a binary tree in which each node contains an integer value.

Find the number of paths that sum to a given value.

The path does not need to start or end at the root or a leaf,
but it must go downwards (traveling only from parent nodes to child nodes).

The tree has no more than 1,000 nodes and the values are in the range -1,000,000 to 1,000,000.
"""


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def pathSum(self, root: TreeNode, target: int) -> int:
        """
        Traverse the tree downward, and try each path.
        Keep during the descent the possible starting sums (in a dictionary for multiplicities).
        On the way up, sum the number of paths.
        """
        if not root:
            return 0

        def visit(prefixes, node: TreeNode) -> int:
            new_prefixes = {key+node.val: count for key, count in prefixes.items()}
            new_prefixes[node.val] = new_prefixes.get(node.val, 0) + 1
            count = new_prefixes.get(target, 0)
            if node.left:
                count += visit(new_prefixes, node.left)
            if node.right:
                count += visit(new_prefixes, node.right)
            return count

        return visit({}, root)

    def pathSum(self, root: TreeNode, target: int) -> int:
        """
        To make things more efficient, you can store only the distance from node to root.
        => This way you can just check whether there is a diff that works.
        """
        if not root:
            return 0

        def visit(prefixes, path_to_root: int, node: TreeNode) -> int:
            path_to_root = path_to_root + node.val
            count = 1 if path_to_root == target else 0
            count += prefixes.get(path_to_root - target, 0)  # path_to_root - prefix == target

            prefixes[path_to_root] = prefixes.get(path_to_root, 0) + 1
            if node.left:
                count += visit(prefixes, path_to_root, node.left)
            if node.right:
                count += visit(prefixes, path_to_root, node.right)
            prefixes[path_to_root] = prefixes.get(path_to_root, 0) - 1
            return count

        return visit({}, 0, root)
