class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


"""
Finding whether there is a cycle:
https://leetcode.com/problems/linked-list-cycle
"""


def has_cycle(head: ListNode) -> bool:
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


"""
Find the exact point where they meet without using extra memory
"""


def find_cycle_start(head: ListNode) -> ListNode:
    """
    To find a cycle, just use the slow and fast pointers.

    To find the place where it connects without using extra-space, let:
    - C be the length of the cycle
    - L be the number of nodes before the cycle
    - K < C the number nodes after start of cycle where 'slow' and 'fast' meet

    We have:
    S = L + C * n1 + K
    F = L + C * n2 + K
    F = 2 * S

    Therefore:
    2 * L + 2 * C * n1 + 2 * K = L + C * n2 + K
    L + 2 * C * n1 + K = C * n2
    L + K = C * (n2 - 2 * n1)

    Therefore
    L + K + 1 is a multiple of C, the length of the cycle

    So when we find a cycle, the algorithm is:
    - 'slow' moves back to 'head'
    - 'fast' starts back from its position, but going at normal speed

    Where they meet is the starting point since fast will have moved
    L + K steps = C from the beginning of the loop
    """
    if not head:
        return None

    slow = fast = head
    while fast is not None:
        slow = slow.next
        fast = fast.next
        if fast is None:
            break
        fast = fast.next
        if slow is fast:
            return find_meeting_point(head, fast)
    return None


def find_meeting_point(slow, fast):
    while slow != fast:
        slow = slow.next
        fast = fast.next
    return slow
