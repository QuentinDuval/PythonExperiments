"""
https://leetcode.com/problems/basic-calculator-ii

Implement a basic calculator to evaluate a simple expression string.

The expression string contains only non-negative integers, +, -, *, / operators and empty spaces.
The integer division should truncate toward zero.
"""


class Solution:
    def calculate(self, s: str) -> int:
        if not s:
            return 0

        numbers = []
        operators = []

        i = 0
        while i < len(s):
            if s[i] in "+-*/":
                operators.append(s[i])
                i += 1
            elif s[i].isdigit():
                j = i + 1
                number = int(s[i])
                while j < len(s) and s[j].isdigit():
                    number = number * 10 + int(s[j])
                    j += 1
                numbers.append(number)

                # pop for * and / in order to deal with precedence
                if operators and operators[-1] in "*/":
                    op = operators.pop()
                    b = numbers.pop()
                    if op == '/':
                        numbers[-1] //= b
                    else:
                        numbers[-1] *= b
                i = j
            else:
                i += 1  # for spaces

        # Beware, it has to be done in that order (not pop)
        total = numbers[0]
        for i in range(len(operators)):
            if operators[i] == '+':
                total += numbers[i + 1]
            else:
                total -= numbers[i + 1]
        return total
