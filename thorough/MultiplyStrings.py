"""
https://practice.geeksforgeeks.org/problems/multiply-two-strings/1

Given two numbers as stings s1 and s2 your task is to multiply them.
You are required to complete the function multiplyStrings which takes two strings s1 and s2 as its only argument
and returns their product as strings.

1 <= T <= 100
1 <= length of s1 and s2 <= 10^3
"""


"""
First technique is like we do as children
- Multiply index 0 of s2 with s1, then index 1 with s1...
- Then do a massive sum of many numbers with carries

  1 2 3
*   2 2
-------
  2 4 6
2 4 6
-------
2 7 0 6

But we can rework this:
- at position 0, you have 1 possible s1[0] * s2[0]
- at position 1, you have 2 possibles s1[0] * s2[1] + s1[1] * s2[0]
=> Iterate from position k = 0 to len(s1) + len(s2) - 1 and keep a carry

Time complexity is O(N*M)
Space complexity is O(N+M)
"""


def multiply_strings(s1: str, s2: str) -> str:
    negative = False
    if s1[0] == '-':
        negative = not negative
        s1 = s1[1:]
    if s2[0] == '-':
        negative = not negative
        s2 = s2[1:]

    s1 = [int(x) for x in reversed(s1)]
    s2 = [int(x) for x in reversed(s2)]

    while s1 and s1[-1] == 0:
        s1.pop()

    while s2 and s2[-1] == 0:
        s2.pop()

    if not s1 or not s2:
        return "0"

    n1 = len(s1)
    n2 = len(s2)

    carry = 0
    result = []
    for k in range(0, n1 + n2 - 1):
        total = carry
        lo = max(0, k - n2 + 1)     # We want k-i < n2 => k-n2 < i
        hi = min(n1, k + 1)         # We want i < n1
        for i in range(lo, hi):
            total += s1[i] * s2[k-i]
        carry, total = divmod(total, 10)
        result.append(total)

    if carry > 0:
        result.append(carry)

    result = "".join(str(x) for x in reversed(result))
    result = "-" + result if negative else result
    return result


print(multiply_strings("123", "22"))
print(multiply_strings("123", "05"))
print(multiply_strings("123", "0"))
print(multiply_strings("123", "-22"))


"""
Improved technique possible:
https://en.wikipedia.org/wiki/Karatsuba_algorithm

s1 = (h1 * X + l1)
s2 = (h2 * X + l2)

s1 * s2 = h1 * h2 * X ^ 2 + (h1 * l2 + h2 * l1) * X + l1 * l2
s1 * s2 = h1 * h2 * X ^ 2 + [(h1 + l1) * (h2 + l2) - h1 * h2 - l1 * l2] * X + l1 * l2

=> Can be done in 3 multiplications (see the repeating terms?)
"""
