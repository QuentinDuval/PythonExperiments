"""
https://practice.geeksforgeeks.org/problems/check-if-linked-list-is-pallindrome/1

Given a singly linked list of size N of integers.
The task is to check if the given linked list is palindrome or not.
"""


from linked.Node import *


def is_palindrome(head: Node):
    """
    Basic solution is to push the numbers visited on a stack, then do a second traversal

    Complexity: O(N) time and O(N) space
    """
    stack = []
    for val in head:
        stack.append(val)

    for val in head:
        if val != stack.pop():
            return False
    return True


def is_palindrome_2(head: Node):
    """
    Optimized version consist in only visiting once by splitting at the middle:
    - collect the first half in a stack
    - visit the second half and pop from the stack

    How to detect the breaking point?
    - A slow pointer will move one by one
    - A fast pointer will move two by two

    For even length collections:
    1 2 3 4 5 6 .
          ^
                ^

    For odd length collections, you have to drop the last element on the stack (if 'next' is None directly):
    1 2 3 4 5 6 7 .
            ^
                  ^

    Complexity: O(N) time and O(N/2) space
    """

    stack = []
    slow = head
    fast = head
    while fast is not None:
        fast = fast.next
        if fast:
            stack.append(slow.val)  # only put the value if fast can at least move once (deal with odd length)
            fast = fast.next
        slow = slow.next

    while slow is not None:
        if slow.val != stack.pop():
            return False
        slow = slow.next

    return True


def is_palindrome_3(head: Node):
    """
    If we can destroy the list while moving through it, we can even change the pointers of the nodes to form a stack
    in place. We could even reconstruct it afterwards.

    Complexity: O(N) time and O(1) space
    """

    stack = Node
    slow = head
    fast = head
    while fast is not None:
        fast = fast.next
        if fast:
            curr = slow
            slow = slow.next
            curr.next = stack
            stack = curr
            fast = fast.next
        else:
            slow = slow.next

    head = slow                         # To reconstruct the list as it was
    while slow is not None:
        if slow.val != stack.val:
            return False
        curr = stack                    # To reconstruct the list as it was
        stack = stack.next
        curr.next = head                # To reconstruct the list as it was
        head = curr                     # To reconstruct the list as it was
        slow = slow.next

    return True


for l in [[1, 2, 3, 2, 1], [1, 2, 2, 1], [1, 2, 3, 1, 2], [1, 2, 1, 2]]:
    for f in [is_palindrome, is_palindrome_2, is_palindrome_3]:
        node = Node.from_list(l)
        print(node.to_list(), "->", f(node))
