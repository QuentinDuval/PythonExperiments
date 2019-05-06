"""
Your task is to implement the function strstr.

The function takes two strings as arguments (s,x) and locates the occurrence of the string x in the string s.
The function returns an integer denoting the first occurrence of the string x i, or -1 if not found.
"""


def strstr(s, x):
    # We want i - j < len(s) => i < len(s) - j => i < len(s) - len(x) + 1
    for i in range(0, len(s) - len(x) + 1):
        match = True
        for j in range(0, len(x)):
            if s[i+j] != x[j]:
                match = False
        if match:
            return i
    return -1


# TODO - implement the better string algorithms

