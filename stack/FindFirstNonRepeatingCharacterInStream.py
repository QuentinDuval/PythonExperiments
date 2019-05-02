"""
https://practice.geeksforgeeks.org/problems/first-non-repeating-character-in-a-stream/0

Given an input stream of N characters consisting only of lower case alphabets.
The task is to find the first non repeating character of the stream, each time a character is inserted to the stream.
If no non repeating element is found print -1.
"""


from collections import deque, defaultdict


def find_first_non_repeating_at_each_step(stream):
    """
    A first implementation is based on a:
    - A counter to detect element already seen
    - A deque to pop from the beginning elements that are seen

    A better implementation could rely on a:
    - A counter to detect element already seen
    - An ordered dictionary to pop elements from anywhere in the queue
    """

    visited = defaultdict(int)
    buffer = deque()
    for val in stream:
        if val not in visited:
            buffer.append(val)
        visited[val] += 1

        while buffer and visited[buffer[0]] > 1:
            buffer.popleft()
        yield buffer[0] if buffer else "-1"


for out in find_first_non_repeating_at_each_step("aabc"):
    print(out)
