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

    '''
    class Iterator:
        def __init__(self, node):
            self.node = node

        def __next__(self):
            if self.node is None:
                raise StopIteration
            val = self.node.val
            self.node = self.node.next
            return val

    def __iter__(self):
        return self.Iterator(self)
    '''

    def __iter__(self):
        curr = self
        while curr is not None:
            yield curr.val
            curr = curr.next

    def to_list(self):
        return list(self)

