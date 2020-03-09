"""
https://leetcode.com/problems/min-stack/
"""


class MinStack:
    """
    Store the mins for every values... but then look at what it looks like
    => get rid of everytime you store the min different than the current val
    """

    def __init__(self):
        self.stack = []
        self.mins = []

    def push(self, x: int) -> None:
        if not self.stack:
            self.stack.append(x)
            self.mins.append(x)
        else:
            self.stack.append(x)
            if x <= self.mins[-1]:
                self.mins.append(x)

    def pop(self) -> None:
        x = self.stack.pop()
        if x == self.mins[-1]:
            self.mins.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.mins[-1]
