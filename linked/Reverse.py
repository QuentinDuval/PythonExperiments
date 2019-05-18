
from linked.Node import *


def reverse(head: Node) -> Node:
    reversed = None
    while head:
        curr = head
        head = head.next
        curr.next = reversed
        reversed = curr
    return reversed


print(reverse(Node.from_list([1, 2, 3])).to_list())
