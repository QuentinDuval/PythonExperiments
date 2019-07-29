"""
https://practice.geeksforgeeks.org/problems/decode-the-string/0

An encoded string (s) is given, the task is to decode it.

The pattern in which the strings were encoded were as follows:
* original string: abbbababbbababbbab
* encoded string : "3[a3[b]1[ab]]".
"""


"""
!!! IMPORTANT !!!
You cannot split by ']' or '[' or search first and last (does not work)
You need to do proper parsing of the input:
- recursively parse the substring
- each parse returns the next start
"""


def decode(encoded: str):
    repeat = [1]
    stack = [""]
    i = 0
    while i < len(encoded):
        if encoded[i].isalpha():
            stack[-1] += encoded[i]
            i += 1
        elif encoded[i].isdigit():
            j = i
            while j < len(encoded) and encoded[j].isdigit():
                j += 1
            repeat.append(int(encoded[i:j]))
            stack.append("")
            i = j + 1 # for the '['
        elif encoded[i] == ']':
            count = repeat.pop()
            substr = stack.pop()
            stack[-1] += count * substr
            i += 1
    return repeat[-1] * stack[-1]


t = int(input())
for _ in range(t):
    encoded = input()
    print(decode(encoded))
