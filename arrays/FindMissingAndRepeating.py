"""
https://practice.geeksforgeeks.org/problems/find-missing-and-repeating/0

Given an unsorted array of size N of positive integers.
One number 'A' from set {1, 2, …N} is missing and one number 'B' occurs twice in array.
Find these two numbers.

Note:
- If you find multiple answers then print the Smallest number found.
- Also, expected solution is O(n) time and constant extra space.
"""


"""
How to approach it?
- We cannot sort because it is O(n log n)
- We cannot use a hash map because it would require extra space
=> We need to find two tricks to summarize the numbers, 2 equations to solve it

But there are other solutions as well:
https://www.geeksforgeeks.org/find-a-repeating-and-a-missing-number/
"""

"""
TWO EQUATIONS

If we sum the entire collection to S, we can compare the sum with N(N+1)/2 (the normal sum)
If we multiply the entire collection to P, we can compare the sum with N! (the normal production)

We have:
    r - m = S - N(N+1)/2
    r / m = P / N!
 
=>
    r = m + S - N(N+1)/2
    m = (S - N(N+1)/2) / (P / N! - 1)

TOO SLOW (due to the big numbers)
"""


def missing_and_repeating_1(nums):
    n = len(nums)

    nums_sum = 0
    nums_prod = 1
    normal_prod = 1
    for i, val in enumerate(nums):
        nums_sum += val
        nums_prod *= val
        normal_prod *= i + 1

    diff_sum = nums_sum - n * (n + 1) / 2
    ratio_prod = nums_prod / normal_prod
    missing = diff_sum / (ratio_prod - 1)
    repeating = missing + diff_sum
    return round(repeating), round(missing)


"""
MARK NEGATIVE INDEX (use MUTABILITY)

Mark as negative the indexes of the elements you visit:
- If an element is already negative, it means the index is the value of the repeated element
- If at the end an element is positive, it is the index of the missing element (second scan)

Another approach (that avoids a second scan) is to sum the collection and make the different with N(N+1)/2.
S - repeating + missing = N(N+1)/2

FAST ENOUGH (smaller numbers)
"""


def missing_and_repeating_2(nums):
    n = len(nums)

    repeating = 0
    nums_sum = 0
    for i, val in enumerate(nums):
        if nums[abs(val) - 1] < 0:
            repeating = abs(val)
        else:
            nums[abs(val) - 1] *= -1
        nums_sum += abs(val)

    diff_sum = nums_sum - n * (n + 1) // 2
    missing = repeating - diff_sum
    return repeating, missing


"""
MARK NEGATIVE INDEX (use MUTABILITY) - variant
Mark as negative the indexes of the elements you visit, and the collect the positive elements
"""


def missing_and_repeating(nums):
    for val in nums:
        nums[abs(val) - 1] *= -1

    incorrects = []
    for i in range(len(nums)):
        if nums[i] > 0:
            incorrects.append(i+1)

    if incorrects[0] in nums:
        return incorrects[0], incorrects[1]
    else:
        return incorrects[1], incorrects[0]


print(missing_and_repeating([1, 3, 3]))
# (3, 2)

print(missing_and_repeating([1, 14, 31, 8, 18, 33, 28, 2, 6, 16, 20, 3, 34, 17, 19, 21, 24, 25, 32, 11, 30, 13, 27, 7, 26, 29, 27, 15, 4, 12, 22, 5, 9, 10]))
# (27, 23)

