"""
https://practice.geeksforgeeks.org/problems/pairwise-swap-elements-of-a-linked-list-by-swapping-data/1

Given a singly linked list of size N. The task is to swap elements pairwise.
"""


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


def pairWiseSwap(head: Node):
    """
    General idea is to move through pairs, and swap them:

    Problem is to:
    1. return the correct 'head'
    2. correctly connect the swap to the next elements
    3. correctly connect the swap to the previous elements

    So we must keep:
    - The head of the new swapped list
    - The tail of the new swapped list
    """

    curr = head
    head = None
    tail = None

    while curr:
        n1, n2 = curr, curr.next    # Take the next two elements
        if n2:
            curr = n2.next          # Move the current cursor 2 elements right
            n1.next = None          # Swap the pair
            n2.next = n1
            if tail:                # Add to the end of the list
                tail.next = n2
            else:
                head = n2           # Or create a new list (if first pass)
            tail = n1               # Update the end of the list
        else:
            if tail:
                tail.next = n1
            else:
                head = n1
            tail = n1
            curr = None

    return head
