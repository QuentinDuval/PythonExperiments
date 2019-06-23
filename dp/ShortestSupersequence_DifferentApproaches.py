from functools import lru_cache
from typing import *


# TODO - shortest string containing both string s1 and s2 as subsequence


def shortest_supersequence_length(s1: str, s2: str) -> int:
    """
    Recursive solution that amends itself quite well to Dynamic Programming
    => O(N ** 2) time and space complexity

    If you are only interested by the length, you can reduce the space complexity to O(N)
    """

    @lru_cache(maxsize=None)
    def visit(i: int, j: int) -> int:
        if i == len(s1) or j == len(s2):
            return max(len(s1) - i, len(s2) - j)

        if s1[i] == s2[j]:
            return 1 + visit(i+1, j+1)
        else:
            return 1 + min(visit(i+1, j), visit(i, j+1))

    return visit(0, 0)


print(shortest_supersequence_length("abc", "daebfc"))
print(shortest_supersequence_length("abcd", "dabc"))


def shortest_supersequence_1(s1: str, s2: str) -> str:
    """
    If we want to return the super sequence, one way is to return the string
    The problem is with the concatenation of strings, which makes the algorithm run in O(N**3)
    """

    @lru_cache(maxsize=None)
    def visit(i: int, j: int) -> str:
        if i == len(s1) or j == len(s2):
            return s1[i:] or s2[j:]

        if s1[i] == s2[j]:
            return s1[i] + visit(i+1, j+1)
        else:
            lhs = visit(i + 1, j)
            rhs = visit(i, j + 1)
            return s1[i] + lhs if len(lhs) <= len(rhs) else s2[j] + rhs

    return visit(0, 0)


def shortest_supersequence_2(s1: str, s2: str) -> str:
    """
    Another solution is to keep track of which decision we made to the final solution and reconstruct it at the end
    With any mechanisms (pointer or next coordinate) that can chain sub-solutions in O(1)
    """
    class Decision:
        def __init__(self, letter, next_decision):
            self.letter = letter
            self.next_decision = next_decision  # Could use indices of the next sub-solution to consider as well...

    @lru_cache(maxsize=None)
    def visit(i: int, j: int) -> Tuple[int, Decision]:
        if i == len(s1) or j == len(s2):
            s = s1[i:] or s2[j:]
            return len(s), Decision(s, None)

        if s1[i] == s2[j]:
            l, d = visit(i+1, j+1)
            return 1 + l, Decision(s1[i], d)
        else:
            lhs = visit(i + 1, j)
            rhs = visit(i, j + 1)
            if lhs[0] <= rhs[0]:
                return 1 + lhs[0], Decision(s1[i], lhs[1])
            else:
                return 1 + rhs[0], Decision(s2[j], rhs[1])

    path = []
    direction = visit(0, 0)[1]
    while direction is not None:
        path.append(direction.letter)
        direction = direction.next_decision
    return "".join(path)


print(shortest_supersequence_1("abc", "daebfc"))
print(shortest_supersequence_1("abcd", "dabc"))
print(shortest_supersequence_2("abc", "daebfc"))
print(shortest_supersequence_2("abcd", "dabc"))
