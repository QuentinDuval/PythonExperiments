
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

    @classmethod
    def from_list(cls, xs):
        if not xs:
            return None

        h = Node(xs[0])
        t = h
        for x in xs[1:]:
            curr = Node(x)
            t.next = curr
            t = curr
        return h

    def to_list(self):
        xs = []
        head = self
        while head:
            xs.append(head.val)
            head = head.next
        return xs


def reverse(head: Node) -> Node:
    reversed = None
    while head:
        curr = head
        head = head.next
        curr.next = reversed
        reversed = curr
    return reversed


print(reverse(Node.from_list([1, 2, 3])).to_list())
