"""
https://leetcode.com/discuss/interview-question/358676/google-remove-extra-edge

Given a binary tree, where an arbitary node has 2 parents i.e two nodes in the tree have the same child.
Identify the defective node and remove an extra edge to fix the tree.

Follow up:
What if the tree is a BST?
"""


from binary_tree.TreeNode import *


def remove_extra_edge(root: TreeNode):
    if not root:
        return

    discovered = {root}
    to_visit = [root]

    def add_to_visit(child):
        if not child:
            return child
        if child in discovered:
            return None
        else:
            discovered.add(child)
            to_visit.append(child)
            return child

    while to_visit:
        node = to_visit.pop()
        node.right = add_to_visit(node.right)
        node.left = add_to_visit(node.left)


# TODO - do the follow up
