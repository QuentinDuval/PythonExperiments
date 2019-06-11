"""
https://leetcode.com/problems/maximum-frequency-stack

Implement FreqStack, a class which simulates the operation of a stack-like data structure.
FreqStack has two functions:
* push(int x), which pushes an integer x onto the stack.
* pop(), which removes and returns the most frequent element in the stack.

If there is a tie for most frequent element, the element closest to the top of the stack is removed and returned.
"""


# TODO - this solution is stupid: it would be great if there was the rule about ties - but we have to keep the stacks
# TODO - so you can just do something with stairs (several piles for different counts)


class FreqNode:
    def __init__(self, value, time):
        self.value = value
        self.stack = [time]

    def __lt__(self, other):
        if len(self.stack) != len(other.stack):
            return len(self.stack) < len(other.stack)
        return self.stack[-1] < other.stack[-1]

    def __eq__(self, other):
        return len(self.stack) == other.stack and self.stack[-1] == other.stack[-1]

    def __repr__(self):
        return repr((self.value, ":", self.stack))


class FreqStack:
    """
    The principle look like a Heap but:
    - we bump the priority by 1 when we add an element that is already there
    - in case of tie, we want to return the element least arrived

    Clearly we need to retrieve an element by its frequency quickly: need some kind of Heap
    Clearly we need to find an element by its value quickly: need for some kind of Hash Map
    Clearly we need to find which element is last arrived: need for some arrival time (but keep the past arrival times as well)
    """

    def __init__(self):
        self.next_index = 0  # the arrival time
        self.values = {}  # value => position in the heap
        self.heap = [None]  # the heap itself

    def push(self, val: int) -> None:
        self.next_index += 1
        node_pos = self.values.get(val, None)
        if node_pos is None:
            node = FreqNode(value=val, time=self.next_index)
            self.heap.append(node)
            self.values[val] = len(self.heap) - 1
            self.swim_up(len(self.heap) - 1)
        else:
            node = self.heap[node_pos]
            node.stack.append(self.next_index)
            self.swim_up(node_pos)

    def pop(self) -> int:
        top_node = self.heap[1]
        top_node.stack.pop()
        if top_node.stack:
            self.sink_down(1)
        else:
            self.swap(1, len(self.heap) - 1)
            self.heap.pop()
            del self.values[top_node.value]
            self.sink_down(1)
        return top_node.value

    def swim_up(self, pos):
        father = pos // 2
        while father > 0 and self.heap[pos] > self.heap[father]:
            self.swap(father, pos)
            pos = pos // 2
            father = pos // 2

    def sink_down(self, pos):
        while 2 * pos < len(self.heap):
            child_pos = 2 * pos
            if 2 * pos + 1 < len(self.heap):
                if self.heap[2 * pos + 1] > self.heap[2 * pos]:
                    child_pos = 2 * pos + 1

            if self.heap[child_pos] > self.heap[pos]:
                self.swap(child_pos, pos)
                pos = child_pos
            else:
                return

    def swap(self, pos1, pos2):
        self.heap[pos1], self.heap[pos2] = self.heap[pos2], self.heap[pos1]
        for pos in pos1, pos2:
            self.values[self.heap[pos].value] = pos

