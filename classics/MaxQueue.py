from collections import deque


class MaxQueue:
    """
    A queue that can always give you the maximum value in O(1)
    """

    def __init__(self):
        self.queue = deque()
        self.maxs = deque()  # Holds (max value, count) pairs - the front always hold the highest value

    def push(self, val):
        self.queue.append(val)
        equal_nb = 1
        while self.maxs and val >= self.maxs[-1][0]:    # Eats the lower values from the back
            if self.maxs[-1][0] == val:
                equal_nb += self.maxs[-1][1]
            self.maxs.pop()
        self.maxs.append((val, equal_nb))

    def pop(self):
        val = self.queue.popleft()
        if val == self.maxs[0][0]:
            count = self.maxs[0][1]
            self.maxs.popleft()
            if count > 1:
                self.maxs.appendleft((val, count - 1))
        return val

    def get_max(self):
        return self.maxs[0][0]
