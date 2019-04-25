"""
https://leetcode.com/problems/add-one-row-to-tree

Given the root of a binary tree, then value v and depth d, you need to add a row of nodes with value v at the given depth d.
The root node is at depth 1.

The adding rule is: given a positive integer depth d, for each NOT null tree nodes N in depth d-1, create two tree nodes
with value v as N's left subtree root and right subtree root.
And N's original left subtree should be the left subtree of the new left subtree root, its original right subtree should be the right subtree of the new right subtree root.
If depth d is 1 that means there is no depth d-1 at all, then create a tree node with value v as the new root of the whole original tree, and the original tree is the new root's left subtree.
"""


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def addOneRow(self, root: TreeNode, v: int, d: int) -> TreeNode:
        def visit(is_left, node, d):
            if d == 1:
                new_node = TreeNode(v)
                if is_left:
                    new_node.left = node
                else:
                    new_node.right = node
                return new_node
            elif not node:
                return node
            else:
                node.left = visit(True, node.left, d - 1)
                node.right = visit(False, node.right, d - 1)
                return node

        return visit(True, root, d)
