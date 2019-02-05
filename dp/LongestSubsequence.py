import functools


"""
Longest common sub sequence
- (i, j) -> (i-1, j), (i, j-1), (i-1, j-1)
"""


def longest_common_sub_sequence(s1, s2):
    """
    First solution based on memoization => O(n^2) time and space
    """

    @functools.lru_cache(maxsize=None)
    def visit(i1, i2):
        if i1 == len(s1) or i2 == len(s2):
            return 0
        elif s1[i1] == s2[i2]:
            return 1 + visit(i1+1, i2+1)
        else:
            return max(visit(i1+1, i2), visit(i1, i2+1))
    return visit(0, 0)


def longest_common_sub_sequence_2(s1, s2):
    """
    Second solution based on bottom up => O(n^2) time and O(n) space
    """

    memo = [0] * (len(s2) + 1)

    for c1 in s1:
        new_memo = [0] * (len(s2) + 1)
        for i2 in range(1, len(memo)):
            if s2[i2-1] == c1:
                new_memo[i2] = 1 + memo[i2-1]
            else:
                new_memo[i2] = max(new_memo[i2-1], memo[i2])
        memo = new_memo

    return memo[-1]


print(longest_common_sub_sequence("abcdef", "accedf"))
print(longest_common_sub_sequence_2("abcdef", "accedf"))


