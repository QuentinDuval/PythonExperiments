"""
https://leetcode.com/problems/my-calendar-i

mplement a MyCalendar class to store your events. A new event can be added if adding the event will not cause a double booking.

Your class will have the method, book(int start, int end). Formally, this represents a booking on the half open interval [start, end), the range of real numbers x such that start <= x < end.

A double booking happens when two events have some non-empty intersection (ie., there is some time that is common to both events.)

For each call to the method MyCalendar.book, return true if the event can be added to the calendar successfully without causing a double booking. Otherwise, return false and do not add the event to the calendar.

Your class will be called like this: MyCalendar cal = new MyCalendar(); MyCalendar.book(start, end)
"""


class MyCalendar:
    """
    Sort the slots reserved by end dates:
    - search for the first interval with end date > new interval start date
    - if this interval starts after the new interval end date, we can insert
    Beats 85% (284 ms)
    """

    def __init__(self):
        self.slots = []

    def book(self, start: int, end: int) -> bool:
        lo = self.next_ending_after(start)
        if lo == len(self.slots):
            self.slots.append((end - 0.5, start))
            return True
        else:
            next_end, next_start = self.slots[lo]
            if end <= next_start:
                self.slots.insert(lo, (end - 0.5, start))
                return True
            return False

    def next_ending_after(self, start):
        lo = 0
        hi = len(self.slots) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            end = self.slots[mid][0]
            if end < start:
                lo = mid + 1
            else:
                hi = mid - 1
        return lo

# Your MyCalendar object will be instantiated and called as such:
# obj = MyCalendar()
# param_1 = obj.book(start,end)