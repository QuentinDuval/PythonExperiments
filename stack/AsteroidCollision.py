"""
https://leetcode.com/problems/asteroid-collision/

We are given an array asteroids of integers representing asteroids in a row.

For each asteroid, the absolute value represents its size, and the sign represents its direction
(positive meaning right, negative meaning left). Each asteroid moves at the same speed.

Find out the state of the asteroids after all collisions.
* If two asteroids meet, the smaller one will explode.
* If both are the same size, both will explode.
* Two asteroids moving in the same direction will never meet.
"""


from typing import List


class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        """
        The ordering of the element in the array counts:
        a negative element on the left of a positive element => they will never meet

        At the end, there can only be:
        * targets moving to the left on the left side of the array returned
        * target moving on the right on the right side of the array returned

        We can do all in O(N) by keeping a stack and popping from it elements that move to the right that collide with
        elements moving to the left that we found by scanning from left to right
        """
        stack = []
        for asteroid in asteroids:
            if asteroid > 0:
                stack.append(asteroid)
            else:
                while stack and stack[-1] > 0 and stack[-1] < abs(asteroid):
                    stack.pop()
                if stack and stack[-1] > 0 and stack[-1] == abs(asteroid):
                    stack.pop()
                elif not stack or stack[-1] < 0:
                    stack.append(asteroid)
        return stack
