from typing import *


def find132pattern(nums: List[int]) -> bool:
    """
    Given a sequence of n integers a1, a2, ..., an, a 132 pattern is a subsequence ai, aj, ak such that i < j < k and ai < ak < aj.
    Design an algorithm that takes a list of n numbers as input and checks whether there is a 132 pattern in the list.

    See great article:
    # https://leetcode.com/problems/132-pattern/discuss/94089/Java-solutions-from-O(n3)-to-O(n)-for-%22132%22-pattern-(updated-with-one-pass-slution)
    """

    """
    Basic O(n^3) solution
    ---------------------
    Test all combinations of i < j < k
    Test whether we have nums[i] < nums[k] < nums[j]
    ---------------------
    Much too slow!
    """

    '''
    n = len(nums)
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                if nums[i] < nums[k] and nums[k] < nums[j]:
                    return True
    return False
    '''

    """
    Optimized O(n^2) solution
    -------------------------
    Consider j as fixed, just keep track of index i of smallest element before
    Test whether we have nums[i] < nums[k] < nums[j]
    ---------------------
    Still too slow!
    """

    '''
    n = len(nums)
    if n < 3:
        return False

    ai = nums[0]
    for j in range(1, n-1):
        for k in range(j+1, n):
            if ai < nums[k] and nums[k] < nums[j]:
                return True
        ai = min(ai, nums[j])
    return False
    '''

    """
    Optimized O(n) solution
    -----------------------
    !!! Scan from the right !!!
    - Precompute the minimums on the left
    - Find a way to easily find the minimum element on the right higher than minimum element on the left
    After all, this is what we are looking for: a peek a, b, c where b is the biggest, and c > a
    """

    '''
    n = len(nums)
    if n < 3:
        return False

    left_mins = [float('inf')]
    for i in range(n-1):
        left_mins.append(min(left_mins[-1], nums[i]))

    rights = []
    for j in reversed(range(1, n)):
        if nums[j] < left_mins[j]:
            continue

        # Find the smallest element on the right that is bigger than smallest element on the left
        while rights and rights[-1] <= left_mins[j]:
            rights.pop()

        # Compare this smallest element with the current number
        if rights and nums[j] > rights[-1]:
            return True

        rights.append(nums[j])

    return False
    '''

    """
    Optimized O(n) solution
    -----------------------
    https://leetcode.com/problems/132-pattern/discuss/94089/Java-solutions-from-O(n3)-to-O(n)-for-%22132%22-pattern-(updated-with-one-pass-slution)
    - 'third' is the greatest element below the greatest element seen so far (and to the left)
    - nums[i] is the left element
    """

    n = len(nums)
    stack_top = n
    third = float('-inf')
    for i in reversed(range(n)):

        # We found a[i] < a[k]
        if nums[i] < third:
            return True

        # Keep increasing a[k] ('third') until we reach nums[i]
        while stack_top < n and nums[i] > nums[stack_top]:
            third = nums[stack_top]
            stack_top += 1

        # Otherwise add nums[i] at the end (necessarily lower than previous elements on stack)
        stack_top -= 1
        nums[stack_top] = nums[i]
    return False
