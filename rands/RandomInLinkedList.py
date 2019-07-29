"""
https://leetcode.com/problems/linked-list-random-node/

Given a singly linked list, return a random node's value from the linked list.
Each node must have the same probability of being chosen.

Follow up:
What if the linked list is extremely large and its length is unknown to you?
Could you solve this efficiently without using extra space?
"""


import random


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:

    def __init__(self, head: ListNode):
        """
        Note that the head is guaranteed to be not null, so it contains at least one node.
        """
        self.head = head
        # self.length = self.get_length()

    def getRandom(self) -> int:
        """
        ALGORITHM IF YOU KNOW THE LENGTH:
        - You have 1/n change to take the first element
        - If you did not took the first element, then 1/(n-1) for the second
        - And so on...
        """

        '''
        n = self.length
        node = self.head
        for i in range(n):
            if 1 == random.randint(1, n - i):
                return node.val
            node = node.next
        return node.val
        '''

        """
        IF YOU DO NOT KNOW THE LENGTH:
        - Think the other way around. If you had a single node, you are sure to select it.
        - If it so happens there is another one after, you have to roll 1/2 for the second to win
        - If it so happens there is another one after, you have to roll 1/3 for the third to win
        - And so one...
        """
        count = 0
        node = self.head
        while node:
            count += 1
            if 1 == random.randint(1, count):
                chosen = node.val
            node = node.next
        return chosen

    def get_length(self):
        length = 0
        node = self.head
        while node:
            length += 1
            node = node.next
        return length
