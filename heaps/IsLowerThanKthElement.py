"""
Given a min-heap 'pq' containing N element, return whether or not the value 'val' is lower or equal than
the Kth value of the heap
"""

from typing import List


"""
The simple approach is to pop K value and check => This approach takes O(K log N)
- This is the only approach available if you need the value of the Kth element of the heap
- But we do not need this value for the problem

The next approach is to do a DFS in the heap and count the number of elements lower than 'val'
Since it is a min-heap, we can avoid going down a heap node if the value is higher than 'val' (all sub-nodes are bigger) 
"""


def is_lower_than_kth_element(heap: List[int], val: int):
    pass # TODO
