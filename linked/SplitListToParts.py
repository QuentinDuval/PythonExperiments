"""
https://leetcode.com/problems/split-linked-list-in-parts/

Given a (singly) linked list with head node root, write a function to split the linked list into k consecutive linked list "parts".

The length of each part should be as equal as possible: no two parts should have a size differing by more than 1. This may lead to some parts being null.

The parts should be in order of occurrence in the input list, and parts occurring earlier should always have a size greater than or equal parts occurring later.

Return a List of ListNode's representing the linked list parts that are formed.

Examples 1->2->3->4, k = 5 // 5 equal parts [ [1], [2], [3], [4], null ]
"""

from typing import *


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def splitListToParts(self, head: ListNode, k: int) -> List[ListNode]:
        parts = []
        size = self.length(head)
        part_size, rest = divmod(size, k)
        for i in range(k):
            part, head = self.split_part(head, part_size + (1 if i < rest else 0))
            parts.append(part)
        return parts

    def split_part(self, head: ListNode, size: int) -> Tuple[ListNode, ListNode]:
        if size == 0:
            return None, head

        prev, tail = None, head
        for _ in range(size):
            prev = tail
            tail = tail.next
        prev.next = None
        return head, tail

    def length(self, head: ListNode) -> int:
        length = 0
        while head:
            length += 1
            head = head.next
        return length
