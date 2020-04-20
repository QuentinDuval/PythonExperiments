class Node:
    def __init__(self, val: int):
        self.val = val
        self.prev = None
        self.next = None


class MyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.count = 0

    def get(self, index: int) -> int:
        if index >= self.count:
            return -1

        node = self.head
        for _ in range(index):
            node = node.next
        return node.val

    def addAtHead(self, val: int) -> None:
        node = Node(val)
        if not self.head:
            self.head = node
            self.tail = node
        else:
            node.next = self.head
            self.head.prev = node
            self.head = node
        self.count += 1

    def addAtTail(self, val: int) -> None:
        node = Node(val)
        if not self.head:
            self.head = node
            self.tail = node
        else:
            node.prev = self.tail
            self.tail.next = node
            self.tail = node
        self.count += 1

    def addAtIndex(self, index: int, val: int) -> None:
        if index > self.count:
            return
        if index == self.count:
            return self.addAtTail(val)
        if index == 0:
            return self.addAtHead(val)

        curr = self.head
        for _ in range(index):
            curr = curr.next
        prev = curr.prev

        node = Node(val)
        node.prev = prev
        node.next = curr
        prev.next = node
        curr.prev = node
        self.count += 1

    def deleteAtIndex(self, index: int) -> None:
        if index >= self.count:
            return

        if index == 0:
            self.head = self.head.next
            if self.head:
                self.head.prev = None
                self.count -= 1
            else:
                self.tail = None
                self.count = 0
            return

        if index == self.count - 1:
            self.tail = self.tail.prev
            if self.tail:
                self.tail.next = None
                self.count -= 1
            else:
                self.head = None
                self.count = 0
            return

        node = self.head
        for _ in range(index):
            node = node.next
        node.prev.next = node.next
        node.next.prev = node.prev
        self.count -= 1

    def to_list(self):
        arr = []
        node = self.head
        while node:
            arr.append(node.val)
            node = node.next
        return arr


def test_1():
    l = MyLinkedList()
    l.addAtHead(1)
    l.addAtTail(3)
    assert 3 == l.get(1)
    l.addAtIndex(1, 2)
    print(l.to_list())
    assert 2 == l.get(1)
    l.deleteAtIndex(1)
    assert 3 == l.get(1)


def test_2():
    l = MyLinkedList()
    l.addAtHead(9)
    assert -1 == l.get(1)
    l.addAtIndex(1, 1)
    l.addAtIndex(1, 7)
    l.deleteAtIndex(1)
    l.addAtHead(7)
    l.addAtHead(4)
    l.deleteAtIndex(1)
    l.addAtIndex(1, 4)
    l.addAtHead(2)
    l.deleteAtIndex(5)


if __name__ == '__main__':
    test_1()
    test_2()


