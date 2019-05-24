"""
https://practice.geeksforgeeks.org/problems/reverse-a-linked-list-in-groups-of-given-size/1
https://leetcode.com/problems/reverse-nodes-in-k-group/
"""


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:

        # Reverse the first k elements
        # - 'group' is the reversed group
        # - 'curr' is the start of the next group
        group = None
        curr = head
        for _ in range(k):
            if curr is None:
                # If the size of the group is not enough, reverse it again
                curr = group
                group = None
                while curr:
                    next_curr = curr.next
                    curr.next = group
                    group = curr
                    curr = next_curr
                return group
            else:
                # Build up the group as reversed
                next_curr = curr.next
                curr.next = group
                group = curr
                curr = next_curr

        # Link the reversed groups together
        # - 'tail' is the next of the reversed group
        tail = group
        while tail.next:
            tail = tail.next
        tail.next = self.reverseKGroup(curr, k)
        return group
