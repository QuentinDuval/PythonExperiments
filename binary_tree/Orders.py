"""
Different traversal orders of a binary tree
"""


from binary_tree.TreeNode import *


def pre_order(node: TreeNode):
    """
    Pre order traversal without using any form of recursion (easy as we visit the node first)
    """
    traversal = []
    to_visit = [node] if node else []
    while to_visit:
        node = to_visit.pop()
        traversal.append(node.val)
        if node.right:
            to_visit.append(node.right)
        if node.left:
            to_visit.append(node.left)
    return traversal


def in_order(node: TreeNode):
    """
    In order traversal without using any form of recursion (needs data structure or continuation)
    """
    traversal = []
    visit_tasks = []

    def visit(node):
        def apply():
            traversal.append(node.val)
        return apply

    def dive(node):
        def apply():
            if node.right:
                visit_tasks.append(dive(node.right))
            visit_tasks.append(visit(node))         # Same idea for post-order, just move this at the end
            if node.left:
                visit_tasks.append(dive(node.left))
        return apply

    if node:
        visit_tasks.append(dive(node))
    while visit_tasks:
        task = visit_tasks.pop()
        task()
    return traversal


tree = TreeNode.from_list([1, [2, [4, [5]]], [3, [6], [7, [], [8]]]])
print(TreeNode.layout(tree))
print(pre_order(tree))
print(in_order(tree))
