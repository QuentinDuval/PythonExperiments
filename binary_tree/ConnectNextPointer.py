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


def connect_better(root: TreeLinkNode):
    """
    Another solution is to:
    - do a DFS starting from the right node
    - keep an array of last seen node at depth 'd' ('d' -> next right node)
    - and connect using this
    """

    to_visit = []
    if root:
        to_visit.append((root, 0))

    last_seen = []

    while to_visit:
        node, depth = to_visit.pop()

        # Check the last seen on the right and connect to it
        if len(last_seen) <= depth:
            last_seen.append(None)
        node.next = last_seen[depth]
        last_seen[depth] = node

        # Put right after for it to be analyzed first
        if node.left:
            to_visit.append((node.left, depth + 1))
        if node.right:
            to_visit.append((node.right, depth + 1))


def connect_bfs(root: TreeLinkNode):
    """
    Another solution is to: do a DBF, stage by stage.
    Simplest and fastest solution in practice, but might use more memory: O(width) vs O(height).
    """
    to_visit = []
    if root:
        to_visit.append(root)

    while to_visit:
        prev = None
        next_to_visit = []
        for curr in to_visit:
            if curr.left:
                next_to_visit.append(curr.left)
            if curr.right:
                next_to_visit.append(curr.right)
            if prev:
                prev.next = curr
            prev = curr
        to_visit = next_to_visit

    return root
