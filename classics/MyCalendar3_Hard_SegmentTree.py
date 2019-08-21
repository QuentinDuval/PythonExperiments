"""
https://leetcode.com/problems/my-calendar-iii/

Implement a MyCalendarThree class to store your events. A new event can always be added.

Your class will have one method, book(int start, int end). Formally, this represents a booking on the half open interval
[start, end), the range of real numbers x such that start <= x < end.

A K-booking happens when K events have some non-empty intersection
(ie., there is some time that is common to all K events.)

For each call to the method MyCalendar.book, return an integer K representing the largest integer such that there
exists a K-booking in the calendar.

Your class will be called like this: MyCalendarThree cal = new MyCalendarThree(); MyCalendarThree.book(start, end)
"""

from collections import defaultdict


class SegmentNode:
    __slots__ = ['max', 'count']

    def __init__(self):
        self.max = 0
        self.count = 0


class LazySegmentTree:
    """
    Using an indexation schema for the nodes (rather than a tree)
    - 1 for the root
    - 2*pos for left child, 2*pos + 1 for right child
    """

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.nodes = defaultdict(SegmentNode)

    def get_root(self):
        return self.nodes[1]

    def add(self, start, end):
        self.add_(start, end, self.min_val, self.max_val, 1)

    def add_(self, start, end, node_start, node_end, node_id):
        if end <= node_start or node_end <= start:
            return self.nodes[node_id].max
        if start <= node_start and node_end <= end:
            node = self.nodes[node_id]
            node.count += 1
            node.max += 1
            return node.max
        else:
            node_mid = node_start + (node_end - node_start) / 2
            left_max = self.add_(start, end, node_start, node_mid, 2 * node_id)
            right_max = self.add_(start, end, node_mid, node_end, 2 * node_id + 1)
            node = self.nodes[node_id]
            node.max = node.count + max(left_max, right_max)
            return node.max


class MyCalendarThree:
    def __init__(self):
        self.segment_tree = LazySegmentTree(min_val=0, max_val=10e9)

    def book(self, start: int, end: int) -> int:
        self.segment_tree.add(start, end)
        return self.segment_tree.get_root().max

# Your MyCalendarThree object will be instantiated and called as such:
# obj = MyCalendarThree()
# param_1 = obj.book(start,end)