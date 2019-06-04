"""
https://leetcode.com/problems/reverse-linked-list-ii

Reverse a linked list from position m to n. Do it in one-pass.

Note: 1 ≤ m ≤ n ≤ length of list.
"""


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def reverseBetween(self, head: ListNode, lo: int, hi: int) -> ListNode:
        """
        TRACKING THE HEAD
        Since lo >= 1, the head might change (it is zero indexed)

        SPECIAL CASES:
        - head might change if lo == 1
        - tail might change if hi == len(list)
        - lo might be equal to hi, in which case, we do not reverse

        SPECIAL CARE:
        - you need to remember the previous element before the reversed part
        - you need to remember the next element after the reversed part
        - you need to remember the tail of the reversed part (one traversal only)
        """

        # Special cases
        if not head or lo >= hi:
            return head

        # Move 'curr' to the beginning of the range to reverse
        # Remember the previous element 'prev'
        prev = None
        curr = head
        for _ in range(lo - 1):
            prev = curr
            curr = curr.next

        # Reversing the sub list
        stack_head = None
        stack_tail = curr
        for _ in range(hi - lo + 1):
            next_curr = curr.next
            curr.next = stack_head
            stack_head = curr
            curr = next_curr

        # Connecting the reversed sub list to the rest
        if prev is not None:
            prev.next = stack_head
        else:
            head = stack_head
        stack_tail.next = curr
        return head
