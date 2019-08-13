"""
https://leetcode.com/problems/expression-add-operators

Given a string that contains only digits 0-9 and a target value, return all possibilities to add binary operators
(not unary) +, -, or * between the digits so they evaluate to the target value.
"""
from typing import List


# TODO - do better


class Solution:
    def addOperators(self, digits: str, target: int) -> List[str]:
        """
        You build an AST, so you can select an operation to perform last... but we do not have parentheses.

        Split the problem in 2:
        - first choose the numbers (splits of positions)
        - then choose the operators (chose the range of multiplication - avoids the precedence rules)

        Beats only 5%...
        """

        def all_numbers(numbers: List[int], pos: int) -> int:
            if pos == len(digits):
                yield list(numbers)
                return

            # Avoid numbers starting with 0 (except 0)
            if digits[pos] == '0':
                numbers.append(0)
                yield from all_numbers(numbers, pos + 1)
                numbers.pop()
                return

            for j in range(pos + 1, len(digits) + 1):
                numbers.append(int(digits[pos:j]))
                yield from all_numbers(numbers, j)
                numbers.pop()

        # Caching is possible here (for (pos, target) and a given list of number)
        def all_operations(memo, numbers: List[int], pos: int, target: int) -> List[str]:
            if pos == len(numbers):
                return [""] if target == 0 else []

            solutions = memo.get((pos, target), None)
            if solutions is not None:
                return solutions

            solutions = []
            current_val = 1
            for j in range(pos, len(numbers)):
                current_val *= numbers[j]
                prefix = "*".join(str(num) for num in numbers[pos:j + 1])
                if j + 1 == len(numbers):
                    if target == current_val:
                        solutions.append(prefix)
                    continue

                # Searching for value X such that current_val + X = target
                for sub_sol in all_operations(memo, numbers, j + 1, target - current_val):
                    solutions.append(prefix + "+" + sub_sol)

                # Searching for value X such that current_val - X = target
                # In such case, you have to revert the solution (+ becomes - and other way around)
                for sub_sol in all_operations(memo, numbers, j + 1, current_val - target):
                    solution = prefix + "-"
                    for c in sub_sol:
                        if c == '+':
                            solution += '-'
                        elif c == '-':
                            solution += '+'
                        else:
                            solution += c
                    solutions.append(solution)

            memo[(pos, target)] = solutions
            return solutions

        possibles = []
        for numbers in all_numbers([], 0):
            possibles.extend(all_operations({}, numbers, 0, target))
        return possibles
