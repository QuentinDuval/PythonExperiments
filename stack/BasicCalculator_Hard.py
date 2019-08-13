"""
https://leetcode.com/problems/basic-calculator

Implement a basic calculator to evaluate a simple expression string.

The expression string may contain open ( and closing parentheses ), the plus + or minus sign -, non-negative integers
and empty spaces.
"""


class Evalulator:
    def __init__(self):
        self.stack = []
        self.op = None

    def add_number(self, number):
        self.stack.append(number)
        if len(self.stack) >= 2:
            b = self.stack.pop()
            a = self.stack.pop()
            self.stack.append(self.op(a, b))
            self.op = None

    def add_operator(self, operator):
        if operator == '+':
            self.op = lambda x, y: x + y
        else:
            self.op = lambda x, y: x - y

    def get_value(self) -> int:
        return self.stack[-1]


class Solution:
    def calculate(self, s: str) -> int:
        """
        You cannot use the double stack algorithm directly since it is based on parentheses
        Here, we might not have parenthese, but precedence rules might apply:
        - * takes precedence over + and - (but we do not have *)
        - () force the precedence rules

        One way would be to parse an AST and then simply evaluate it... but seems overkill
        (plus actually all the difficulty is in building the tree, we could evaluate directly)

        Instead, we should evaluate eagerly whenever we can, but recurse inside parentheses.
        - use a stack for the parenthesis (push one when open paren, pop one when close parens)
        - each stack contains an evaluation context, that also has a stack (2 integers, 1 operator)
        """

        i = 0
        parens = [Evalulator()]
        while i < len(s):
            if s[i] == '(':
                parens.append(Evalulator())
                i += 1
            elif s[i] == ')':
                evaluator = parens.pop()
                parens[-1].add_number(evaluator.get_value())
                i += 1
            elif s[i].isdigit():
                number = int(s[i])
                j = i + 1
                while j < len(s) and s[j].isdigit():
                    number = number * 10 + int(s[j])
                    j += 1
                parens[-1].add_number(number)
                i = j
            elif s[i] in {'+', '-'}:
                parens[-1].add_operator(s[i])
                i += 1
            else:
                i += 1  # for spaces
        return parens[-1].get_value()

