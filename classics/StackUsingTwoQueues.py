"""
https://practice.geeksforgeeks.org/problems/stack-using-two-queues/1
"""


from collections import deque


class Stack:
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()

    def push(self, x):
        self.q1.append(x)
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())

    def pop(self):
        if len(self.q1) == 0:
            return None
        val = self.q1.popleft()
        while len(self.q2) > 1:
            self.q1.append(self.q2.popleft())
        self.q1, self.q2 = self.q2, self.q1
        return val
