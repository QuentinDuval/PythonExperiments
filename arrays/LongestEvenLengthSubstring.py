"""
https://practice.geeksforgeeks.org/problems/longest-even-length-substring/0/?ref=self

For given string ‘str’ of digits, find length of the longest substring of ‘str’, such that the length of the substring
is 2k digits and sum of left k digits is equal to the sum of right k digits.
"""


def longest_even_length(digits):
    """
    The naive algorithm is O(N ** 3): testing for each i < j and summing between them.

    We can improve on this algorithm by:
    - pre-computing the partial sums in O(N**2)
    - trying for each i < j the partial sums of [i, j) == [j, j+j-i)
    """

    # TODO - We can avoid the O(N**2) storage by computing prefix sums (and substracting to get the sub-array range).

    partial_sums = {}
    for i in range(len(digits)):
        partial_sums[(i, i)] = digits[i]
        for j in range(i + 1, len(digits)):
            partial_sums[(i, j)] = partial_sums[(i, j - 1)] + digits[j]

    longest = 0
    for i in range(len(digits)):
        for width in range(2, len(digits) - i + 1, 2):
            k = width // 2
            if partial_sums[(i, i + k - 1)] == partial_sums[(i + k, i + 2 * k - 1)]:
                longest = max(longest, width)
    return longest


t = int(input())
for _ in range(t):
    digits = [int(x) for x in input()]
    print(longest_even_length(digits))
