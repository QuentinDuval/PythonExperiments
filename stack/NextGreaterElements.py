from typing import List


"""
https://leetcode.com/problems/next-greater-element-ii/

Given a circular array (the next element of the last element is the first element of the array), print the Next Greater Number for every element.

The Next Greater Number of a number x is the first greater number to its traversing-order next in the array,
which means you could search circularly to find its next greater number. If it doesn't exist, output -1 for this number.
"""


class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        """
        If you are looking for the next greatest element, the simplest way to do it is to use a stack.
        Go from right to left:
        - pop from the stack until you find a greater element (store its index)
        - put the number on top of the stack

        It leaves some elements without a greater element on the right possibly:
        - Go for another round for those elements without greater elements

        Complexity: O(N)
        """
        nexts = []
        greater = [-1] * len(nums)

        for _ in range(2):
            for i in reversed(range(len(nums))):
                num = nums[i]
                # Replace all the elements that are lower
                while nexts and nexts[-1] <= num:
                    nexts.pop()
                if nexts and greater[i] != 1:
                    greater[i] = nexts[-1]
                nexts.append(num)

        return greater