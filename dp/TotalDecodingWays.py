"""
https://practice.geeksforgeeks.org/problems/total-decoding-messages/0

A top secret message containing letters from A-Z is being encoded to numbers using the following mapping:
'A' -> 1
'B' -> 2
...
'Z' -> 26

You are an FBI agent. You have to determine the total number of ways that message can be decoded.
"""


possibles = set(str(x) for x in range(1, 27))


"""
Dynamic programming approach:
- Because we can advance by 1 or 2 at each step, there are overlapping solutions
- There are also sub-structure problems (left to right sub-structure)

Number of sub-problems: O(N)
Time complexity is O(N)
Space complexity is O(N) in top-bottom
"""


def total_decoding_count(encrypted: str) -> int:
    memo = {}

    def recur(i):
        if i in memo:
            return memo[i]
        if i == len(encrypted):
            return 1

        total = 0
        if encrypted[i] in possibles:
            total += recur(i+1)
        if i + 1 < len(encrypted) and encrypted[i:i+2] in possibles:
            total += recur(i+2)
        memo[i] = total
        return total

    return recur(0)


"""
Improved solution where space complexity is O(1)
(Like Fibonacci! See how similar it is)
"""


def total_decoding_count(encrypted: str) -> int:
    n = len(encrypted)
    n_plus_2 = 0
    n_plus_1 = 1

    for i in reversed(range(n)):
        total = 0
        if encrypted[i] in possibles:
            total += n_plus_1
        if i + 1 < n and encrypted[i:i + 2] in possibles:
            total += n_plus_2
        n_plus_1, n_plus_2 = total, n_plus_1

    return n_plus_1


"""
Test client for geeks for geeks
"""


t = int(input())
for _ in range(t):
    n = int(input())
    encrypted = input()[:n]
    print(total_decoding_count(encrypted))
