"""
https://practice.geeksforgeeks.org/problems/check-if-string-is-rotated-by-two-places/0
"""


def is_rotated_by(a, b, shift):
    for i in range(len(a)):
        j = (i + shift) % len(a)
        if a[i] != b[j]:
            return False
    return True


def is_rotated(a, b):
    if len(a) != len(b):
        return False
    if len(a) < 2:
        return False

    # The big mistake here is to try to deduce the shift by doing a[2] == b[0] or a[0] == b[2]
    # Indeed, both could be true... and testing one possibility would make you fail
    return is_rotated_by(a, b, 2) or is_rotated_by(a, b, -2)
