"""
Different traversal orders of a binary tree
"""


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    @classmethod
    def from_list(cls, xs):
        if type(xs) is not list:
            return cls(xs)
        elif not xs:
            return None
        else:
            node = cls(xs[0])
            node.left = cls.from_list(xs[1]) if len(xs) >= 2 else None
            node.right = cls.from_list(xs[2]) if len(xs) >= 3 else None
            return node

    @classmethod
    def depth(cls, node):
        max_depth = 0
        to_visit = [(node, 1)] if node else None
        while to_visit:
            node, depth = to_visit.pop()
            max_depth = max(depth, max_depth)
            if node.right:
                to_visit.append((node.right, depth + 1))
            if node.left:
                to_visit.append((node.left, depth + 1))
        return max_depth

    @classmethod
    def layout(cls, root):
        """
        A layout like this:

              1
            2     3
          4     6     7
        5           8

        Always shift one compared to the element below
        """
        max_depth = cls.depth(root)
        lines = ["" for _ in range(max_depth)]

        def visit(node, depth, shift):
            if node.left:
                shift = visit(node.left, depth + 1, shift)

            lines[depth] += (shift - len(lines[depth])) * " " + str(node.val)
            shift = len(lines[depth]) + 1
            if node.right:
                shift = visit(node.right, depth + 1, shift)
            return shift

        visit(root, depth=0, shift=0)
        return "\n".join(lines)


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
