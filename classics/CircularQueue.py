"""
https://leetcode.com/problems/design-circular-queue
"""


# Your MyCircularQueue object will be instantiated and called as such:
# obj = MyCircularQueue(k)
# param_1 = obj.enQueue(value)
# param_2 = obj.deQueue()
# param_3 = obj.Front()
# param_4 = obj.Rear()
# param_5 = obj.isEmpty()
# param_6 = obj.isFull()


class MyCircularQueue:

    def __init__(self, max_size: int):
        self.queue = [0] * max_size
        self.front = 0
        self.rear = -1
        self.size = 0

    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False

        self.rear = self._next_index(self.rear)
        self.queue[self.rear] = value
        self.size += 1
        return True

    def deQueue(self) -> bool:
        if self.isEmpty():
            return False

        self.front = self._next_index(self.front)
        self.size -= 1
        return True

    def Front(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[self.front]

    def Rear(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[self.rear]

    def isEmpty(self) -> bool:
        return self.size == 0

    def isFull(self) -> bool:
        return self.size == len(self.queue)

    def _next_index(self, index, jump=1):
        if index + jump >= len(self.queue):
            return 0
        return index + jump
