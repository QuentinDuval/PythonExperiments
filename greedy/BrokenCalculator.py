"""
https://leetcode.com/problems/broken-calculator/

On a broken calculator that has a number showing on its display, we can perform two operations:
- Double: Multiply the number on the display by 2, or;
- Decrement: Subtract 1 from the number on the display.

Initially, the calculator is displaying the number X.
Return the minimum number of operations needed to display the number Y.
"""


class Solution:
    def brokenCalc(self, x: int, y: int) -> int:
        """
        The obvious brute force method is to do a kind of BFS
        Timeout! (exponential in nature)
        """

        '''
        count = 0
        current = { x }
        while y not in current:
            count += 1
            next_current = { x - 1 for x in current if x > 0 }
            next_current |= { x * 2 for x in current }
            current = next_current
        return count
        '''

        """
        So you need to think a bit:
        - if 'x' is higher than 'y' then the result is 'y - x' (no need to multiply)
        - if 'x' is lower than 'y' then exponentiation should prevail, but how?

        Example of 1 => 100

        1, 2, 4, 8, 16, 32, 64, 128..100 (7 + 28 moves)
        1, 2, 4, 8, 16, 32, 64..50, 100 (7 + 14 moves)
        1, 2, 4, 8, 16, 32..25, 50, 100 (7 + 7 moves)
        1, 2, 4, 8, 16..13, 26..25, 50, 100 (7 + 4 moves)
        1, 2, 4, 8..7, 14..13, 26..25, 50, 100 (7 + 3 moves) => BEST

        The goal is to overshoot the target with a fix number of multiplications
        (but the least amount of overshooting)
        - Example of going too down: 1, 2, 4, 8..6, 12, 24, 48, 96, ...
        - Example of going right down: 1, 2, 4, 8..7, 14, 28, 56, 112
        => So keep the amount of multiplications fixed, and overshoot the least possible (counter productive otherwise)
        => Do this at each stage, going from left to right (maximum impact to minimum impact)
        """

        if x > y:
            return x - y

        lower_bounds = [y]
        multiples = [x]
        while multiples[-1] < y:
            multiples.append(multiples[-1] * 2)
            lower_bounds.append(lower_bounds[-1] / 2)
        lower_bounds = lower_bounds[::-1]

        decr_count = 0
        mult_count = len(multiples) - 1
        for i in range(len(multiples)):
            diff = int(multiples[i] - lower_bounds[i])
            if diff > 0:
                decr_count += diff
                multiples[i] -= diff
                for j in range(i + 1, len(multiples)):
                    multiples[j] = multiples[j - 1] * 2
        return decr_count + mult_count
