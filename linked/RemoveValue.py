"""
https://leetcode.com/problems/remove-linked-list-elements

Remove all elements from a linked list of integers that have value val.
"""


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        while head and head.val == val:
            curr = head
            head = head.next
            del curr

        if not head:
            return head

        prev = head
        curr = head.next
        while curr:
            if curr.val == val:
                toDel = curr
                curr = curr.next
                prev.next = curr
                del toDel
            else:
                prev = curr
                curr = curr.next
        return head
