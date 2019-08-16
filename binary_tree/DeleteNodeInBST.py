"""
https://leetcode.com/problems/delete-node-in-a-bst
"""


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        """
        Three cases:
        - the node to delete has only a left child: make it the node
        - the node to delete has only a right child: make it the node
        - the node to delete has two children:
            - search the right-most node on the left subtree, delete it
            - take its value and put it in place of the current node
            - this right-most node has only one child (cause rightmost) => easy

        Beats 93%
        """

        def rightmost(node: TreeNode) -> TreeNode:
            while node.right:
                node = node.right
            return node

        def delete_node(node: TreeNode, key) -> TreeNode:

            # Finding then node
            if not node:
                return None
            elif key < node.val:
                node.left = delete_node(node.left, key)
                return node
            elif key > node.val:
                node.right = delete_node(node.right, key)
                return node

            # Deleting the node
            if not node.left:
                return node.right
            elif not node.right:
                return node.left
            else:
                to_delete = rightmost(node.left)
                node.val = to_delete.val
                node.left = delete_node(node.left, node.val)
                return node

        return delete_node(root, key)

