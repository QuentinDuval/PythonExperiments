"""
https://leetcode.com/problems/validate-stack-sequences/

Given two sequences pushed and popped with distinct values, return true if and only if this could have been the result
of a sequence of push and pop operations on an initially empty stack.
"""


from typing import List


class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        """
        Do an example to see what kind of permutations are allowed:

        [1, 2, 3]
        =>
        [1, 2, 3] (push and pop each time)
        [1, 3, 2] (pop 1 immediately)
        [2, 1, 3] (pop 2 then 1, then add 3)
        [2, 3, 1] (pop 3 then add 3)
        [3, 1, 2] IMPOSSIBLE
        [3, 2, 1] (pop all at the end)

        What makes [3, 1, 2] impossible is that 3 cannot be followed by sorted lower numbers.
        We see the same things in the examples:
        - [1,2,3,4,5] to [4,5,3,2,1] is fine
        - [1,2,3,4,5] to [4,3,5,1,2] is not because of [1, 2]
        => So the algorithm is simple, you can check this property.
        """

        """
        Other approach? Just try the algorithm:
        - push on a stack as long as the next element to pop does not show
        - pop the element on the stack
        If you run out of element to push, then return False
        """
        if len(pushed) != len(popped):
            return False

        push_i = 0
        pop_i = 0
        stack = []
        while pop_i < len(popped):
            while not (stack) or stack[-1] != popped[pop_i]:
                if push_i < len(pushed):
                    stack.append(pushed[push_i])
                    push_i += 1
                else:
                    return False
            stack.pop()
            pop_i += 1
        return True
