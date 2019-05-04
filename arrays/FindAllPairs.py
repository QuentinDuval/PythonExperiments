"""
Given two unsorted arrays A of size N and B of size M of distinct elements,
the task is to find all pairs from both arrays whose sum is equal to X.
"""


def find_all_pairs(xs, ys, target):
    """
    Scan each of the elements in xs (or ys), and for each 'x'
    look in the other collection the 'target - x'.

    There are many ways to do this efficiently:
    - sort the second collection and do a binary search in O(log n)
    - hash the second collection and do a search in O(1)

    Note that it works well because the inputs contain DISTINCT elements
    If not, we would have to take into account duplicates
    """
    pairs = []
    ys = set(ys)
    for x in xs:
        if target - x in ys:
            pairs.append((x, target - x))
    return pairs
