from binary_tree.TreeNode import *


def is_identical(lhs: TreeNode, rhs: TreeNode):
    """
    Non-recursive implementation (recursive implementation is trivial)
    """
    to_check = [(lhs, rhs)]
    while to_check:
        lhs, rhs = to_check.pop()
        if lhs is None and rhs is None:
            continue
        if lhs is None or rhs is None:
            return False
        if lhs.val != rhs.val:
            return False
        to_check.append((lhs.right, rhs.right))
        to_check.append((lhs.left, rhs.left))

    return True
