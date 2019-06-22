import pytest
from typing import List


def cache(f):
    memo = {}

    def wrapped(*args):
        if args in memo:
            return memo[args]
        res = f(*args)
        memo[args] = res
        return res

    return wrapped


def longest_increasing_subsequence(nums: List[int]) -> int:

    @cache
    def longest_with(i: int) -> int:
        count = 0
        for next_pos in range(i + 1, len(nums)):
            if nums[next_pos] >= nums[i]:
                count = max(count, longest_with(next_pos))
        return 1 + count

    return max(longest_with(i) for i in range(len(nums)))


# TODO - all of this should work but it does not work (pytest does not found the test_ functions nor TestClasses)...


class TestLongestIncreasingSubsequence:

    def test_increasing_list(self):
        assert 3 == longest_increasing_subsequence([1, 2, 3])

    def test_decreasing_list(self):
        assert 1 == longest_increasing_subsequence([3, 2, 1])

    def test_monotonous_list(self):
        assert 3 == longest_increasing_subsequence([2, 2, 2])


def test_subsequence():
    assert 4 == longest_increasing_subsequence([5, 1, 4, 3, 2, 3, 4])


pytest.main()
