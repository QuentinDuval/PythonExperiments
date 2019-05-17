"""
https://practice.geeksforgeeks.org/problems/delete-without-head-pointer/1

You are given a pointer/reference to a node to be deleted in a linked list of size N. The task is to delete the node.
Pointer/reference to head node is not given.

You may assume that the node to be deleted is not the last node.
"""


class Node:
    def __init__(self, val):
        self.val = val
        self.next = None


def delete(node: Node):
    """
    Since we do not have the head pointer, it is not possible to delete the node.

    The tick is therefore to take the value of the next node, and then skip this node
    !!! In C++, do not forget to delete the skipped node !!!
    """
    next = node.next
    node.val = next.val
    node.next = next.next
    del next
