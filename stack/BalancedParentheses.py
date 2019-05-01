"""
https://practice.geeksforgeeks.org/problems/parenthesis-checker/0
"""


def is_balanced(expression):
    stack = []
    for c in expression:
        if c in "([{":
            stack.append(c)
        elif not stack:
            return False
        else:
            last = stack.pop()
            if c == ")" and last == "(":
                continue
            if c == "}" and last == "{":
                continue
            if c == "]" and last == "[":
                continue
            return False
    return len(stack) == 0


test_nb = int(input())
for _ in range(test_nb):
    expression = input()
    if is_balanced(expression):
        print("balanced")
    else:
        print("not balanced")
