"""
https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii

Populate each next pointer to point to its next right node. If there is no next right node, set to NULL.
Initially, all next pointers are set to NULL.
"""


class TreeLinkNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.next = None


def connect(root: TreeLinkNode):
    """
    At each node, check if it has a 'left' and 'right', in which case link them.
    But for the right child, we also need to link it to a potential sub-tree of its right.
    To do this efficiently, we can make use of the 'next' pointer of the parent.
    """
    if not root:
        return

    toVisit = [root]
    while toVisit:
        node = toVisit.pop()

        # Connect left to right
        if node.left and node.right:
            node.left.next = node.right

        # Connect rightmost node to next tree
        if node.right or node.left:
            right = node.right or node.left
            right_parent = node.next
            while right_parent:
                right.next = right_parent.left or right_parent.right
                if right.next:
                    break
                right_parent = right_parent.next

        # Depth first search
        if node.left:
            toVisit.append(node.left)
        if node.right:
            toVisit.append(node.right)
