"""
https://leetcode.com/problems/parse-lisp-expression
"""

from typing import *


class Solution:
    def evaluate(self, expr: str) -> int:
        val, pos = self.parse_expr({}, expr, 0)
        return val

    def parse_expr(self, env: Dict[str, int], expr: str, pos: int) -> Tuple[int, int]:
        if expr[pos] == '(':
            return self.parse_form(env, expr, pos + 1)
        if expr[pos].isdigit() or expr[pos] == '-':
            return self.parse_digit(env, expr, pos)
        if expr[pos].islower():
            var_name, pos = self.parse_var(env, expr, pos)
            return env[var_name], pos

    def parse_digit(self, env: Dict[str, int], expr: str, pos: int) -> Tuple[int, int]:
        end = pos + 1
        while end < len(expr) and expr[end].isdigit():
            end += 1
        value = int(expr[pos:end])
        return value, end

    def parse_var(self, env: Dict[str, int], expr: str, pos: int) -> Tuple[str, int]:
        end = pos
        while end < len(expr) and (expr[end].isalpha() or expr[end].isdigit()):
            end += 1
        var_name = expr[pos:end]
        return var_name, end

    def parse_function(self, env: Dict[str, int], expr: str, pos: int, f) -> Tuple[int, int]:
        pos = self.skip_spaces(expr, pos)
        a, pos = self.parse_expr(env, expr, pos)
        pos = self.skip_spaces(expr, pos)
        b, pos = self.parse_expr(env, expr, pos)
        res = f(a, b)
        pos = self.skip_spaces(expr, pos)
        return res, pos + 1  # add 1 to skip parenthesis

    def parse_assign(self, env: Dict[str, int], expr: str, pos: int) -> Tuple[bool, int]:
        pos = self.skip_spaces(expr, pos)
        if not expr[pos].islower():
            return False, pos

        var_name, end_pos = self.parse_var(env, expr, pos)
        end_pos = self.skip_spaces(expr, end_pos)
        if expr[end_pos] == ')':
            return False, pos

        val, pos = self.parse_expr(env, expr, end_pos)
        env[var_name] = val
        return True, pos

    def parse_let(self, env: Dict[str, int], expr: str, pos: int) -> Tuple[int, int]:
        new_env = dict(env)
        while True:
            good, pos = self.parse_assign(new_env, expr, pos)
            if not good:
                break
        val, pos = self.parse_expr(new_env, expr, pos)
        return val, pos + 1  # add 1 to skip closing parenthesis

    def parse_form(self, env: Dict[str, int], expr: str, pos: int) -> Tuple[int, int]:
        if expr[pos:pos + 3] == 'let':
            return self.parse_let(env, expr, pos + 3)
        if expr[pos:pos + 3] == 'add':
            return self.parse_function(env, expr, pos + 3, lambda a, b: a + b)
        if expr[pos:pos + 4] == 'mult':
            return self.parse_function(env, expr, pos + 4, lambda a, b: a * b)

    def skip_spaces(self, expr: str, pos: int) -> int:
        while pos < len(expr) and expr[pos] == ' ':
            pos += 1
        return pos
