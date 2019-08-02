"""
https://leetcode.com/problems/add-two-numbers-ii

You are given two non-empty linked lists representing two non-negative integers.
The most significant digit comes first and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Follow up:
What if you cannot modify the input lists? In other words, reversing the lists is not allowed.
"""

from linked.Node import Node


class Solution:
    def addTwoNumbers(self, l1: Node, l2: Node) -> Node:
        """
        Solution using a reverse of the list
        (Other solution consist in using a stack if you cannot modify the input)
        """
        l1 = self.reverse(l1)
        l2 = self.reverse(l2)

        result = None
        carry = 0
        while l1 or l2:
            digit = carry
            if l1:
                digit += l1.val
                l1 = l1.next
            if l2:
                digit += l2.val
                l2 = l2.next
            carry, digit = divmod(digit, 10)
            result = self.add_digit(result, digit)
        if carry:
            result = self.add_digit(result, carry)
        return result

    def add_digit(self, node, digit):
        new_node = Node(digit)
        new_node.next = node
        return new_node

    def reverse(self, l):
        prev, curr = None, l
        while curr:
            next_curr = curr.next
            curr.next = prev
            prev, curr = curr, next_curr
        return prev
