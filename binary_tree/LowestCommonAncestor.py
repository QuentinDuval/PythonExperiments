"""
https://practice.geeksforgeeks.org/problems/lowest-common-ancestor-in-a-bst/1

Given a Binary Search Tree and 2 nodes value n1 and n2, your task is to find the lowest common ancestor(LCA) of the two nodes.
"""


from binary_tree.TreeNode import *


def LCA(root: TreeNode, a: int, b: int):
    """
    We know that the tree is a BST: so we can avoid searching everywhere.

    First solution is to look for 'a' and 'b' using BST search.
    - Trace all the elements to reach them
    - Find the last element common to both

    But we can do better: the lowest ancestor is necessarily the one
    for which we fork the search for 'a' and 'b'.
    => it is the first one for which 'a' <= element <= 'b'

    No need for recursion here: we only visit one side of the tree.
    """
    curr = root
    a, b = min(a, b), max(a, b)         # Do not miss this ! Otherwise it does not converge.
    while curr:
        if a <= curr.val <= b:
            return curr
        if a < curr.val:
            curr = curr.left
        else:
            curr = curr.right
    return None
