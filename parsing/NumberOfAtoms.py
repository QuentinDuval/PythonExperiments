"""
https://leetcode.com/problems/number-of-atoms/

Given a chemical formula (given as a string), return the count of each atom.

An atomic element always starts with an uppercase character, then zero or more lowercase letters, representing the name.

1 or more digits representing the count of that element may follow if the count is greater than 1. If the count is 1,
no digits will follow. For example, H2O and H2O2 are possible, but H1O2 is impossible.

Two formulas concatenated together produce another formula. For example, H2O2He3Mg4 is also a formula.

A formula placed in parentheses, and a count (optionally added) is also a formula. For example, (H2O2) and (H2O2)3 are
formulas.

Given a formula, output the count of all elements as a string in the following form: the first name (in sorted order),
followed by its count (if that count is more than 1), followed by the second name (in sorted order), followed by its
count (if that count is more than 1), and so on.
"""


from collections import defaultdict


class Solution:
    def countOfAtoms(self, s: str) -> str:

        def parse_number(start):
            end = start
            while end < len(s) and s[end].isdigit():
                end += 1
            mult = int(s[start:end]) if end > start else 1
            return mult, end

        i = 0
        stack = [defaultdict(int)]

        while i < len(s):
            if s[i] == '(':
                stack.append(defaultdict(int))
                i += 1
            elif s[i] == ')':
                mult, i = parse_number(i + 1)
                counter = stack.pop()
                for elem, count in counter.items():
                    stack[-1][elem] += mult * count
            elif s[i].isupper():
                j = i + 1
                while j < len(s) and s[j].islower():
                    j += 1
                elem = s[i:j]
                mult, i = parse_number(j)
                stack[-1][elem] += mult

        return self.render_elements(stack[-1])

    def render_elements(self, counter):
        return "".join(self.render(elem, count) for elem, count in sorted(counter.items()))

    def render(self, elem, count):
        if count == 1:
            return elem
        else:
            return elem + str(count)
