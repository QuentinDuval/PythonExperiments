
"""
Finding whether there is a cycle:
https://leetcode.com/problems/linked-list-cycle
"""


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


def has_cycle(head):
    if not head:
        return False

    slow = head
    fast = head.next
    while fast is not None:
        if slow is fast:
            return True
        slow = slow.next
        fast = fast.next
        if fast is not None:
            fast = fast.next
        else:
            return False
    return False
