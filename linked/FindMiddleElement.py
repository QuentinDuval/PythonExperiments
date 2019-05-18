"""
https://practice.geeksforgeeks.org/problems/finding-middle-element-in-a-linked-list/1
"""


from linked.Node import *


def find_middle(head: Node) -> Node:
    """
    Important point is that we want the middle (not the start of the second half):
    Therefore, 'slow' should only advance if 'fast' could advance twice.
    """
    slow = head
    fast = head
    while fast is not None:
        fast = fast.next
        if fast:
            fast = fast.next
            slow = slow.next
    return slow
