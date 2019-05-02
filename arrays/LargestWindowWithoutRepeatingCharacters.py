"""
Given a string, return the largest substring that does not contain repeating characters
"""


def largest_window(s):
    best_start = 0
    best_end = -1

    start = 0
    found = set()
    for end in range(len(s)):
        while s[end] in found:
            found.remove(s[start])
            start += 1
        found.add(s[end])
        if end - start > best_end - best_start:
            best_start = start
            best_end = end

    return s[best_start:best_end+1]


print(largest_window("abcdcaefa"))
