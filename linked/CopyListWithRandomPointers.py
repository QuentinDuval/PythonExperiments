"""
Copy a list that contains random pointers to other elements of the list
https://leetcode.com/problems/copy-list-with-random-pointer
"""


class RandomListNode(object):
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None


def copyRandomList(head: RandomListNode) -> RandomListNode:
    """
    We need a mapping of the pointers, in order to, in a second pass,
    rewire the 'random' nodes to the correct nodes.

    The algorithm proceeds in two passes:
    - First deep copy the list with the next pointers, keeping the mapping
    - Then do a second pass, mapping the random pointers correctly
    """
    if not head:
        return None

    clone = RandomListNode(head.label)
    mapping = {head: clone}

    clone_tail = clone
    tail = head
    while True:
        tail = tail.next
        if tail is None:
            break

        node = RandomListNode(tail.label)
        mapping[tail] = node
        clone_tail.next = node
        clone_tail = node

    curr = clone
    while head is not None:
        if head.random is not None:
            curr.random = mapping[head.random]
        head = head.next
        curr = curr.next

    return clone
