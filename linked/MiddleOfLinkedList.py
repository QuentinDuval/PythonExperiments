"""
https://leetcode.com/problems/middle-of-the-linked-list

Given a non-empty, singly linked list with head node head, return a middle node of linked list.

If there are two middle nodes, return the second middle node.
"""


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        slow = fast = head
        while fast:
            fast = fast.next
            if fast:
                slow = slow.next
                fast = fast.next
        return slow
