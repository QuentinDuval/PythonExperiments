"""
https://leetcode.com/problems/design-circular-deque
"""


# Your MyCircularDeque object will be instantiated and called as such:
# obj = MyCircularDeque(k)
# param_1 = obj.insertFront(value)
# param_2 = obj.insertLast(value)
# param_3 = obj.deleteFront()
# param_4 = obj.deleteLast()
# param_5 = obj.getFront()
# param_6 = obj.getRear()
# param_7 = obj.isEmpty()
# param_8 = obj.isFull()


class MyCircularDeque:

    def __init__(self, max_size: int):
        self.deque = [0] * max_size
        self.size = 0
        self.front = 0
        self.rear = -1

    def _move(self, index, jump=1):
        index = index + jump
        if index < 0:
            return index + len(self.deque)
        if index >= len(self.deque):
            return index - len(self.deque)
        return index

    def insertFront(self, value: int) -> bool:
        if self.isFull():
            return False

        self.front = self._move(self.front, -1)
        self.deque[self.front] = value
        self.size += 1
        return True

    def deleteFront(self) -> bool:
        if self.isEmpty():
            return False

        self.front = self._move(self.front, 1)
        self.size -= 1
        return True

    def insertLast(self, value: int) -> bool:
        if self.isFull():
            return False

        self.rear = self._move(self.rear, 1)
        self.deque[self.rear] = value
        self.size += 1
        return True

    def deleteLast(self) -> bool:
        if self.isEmpty():
            return False

        self.rear = self._move(self.rear, -1)
        self.size -= 1
        return True

    def getFront(self) -> int:
        if self.isEmpty():
            return -1
        return self.deque[self.front]

    def getRear(self) -> int:
        if self.isEmpty():
            return -1
        return self.deque[self.rear]

    def isEmpty(self) -> bool:
        return self.size == 0

    def isFull(self) -> bool:
        return self.size == len(self.deque)
