"""
This does not work:
- for "mississipie", it outputs "mipie" instead of "mpie"
- the reason is that we do not want to remove the "i" as fast
"""

'''
def remove_adjacent_duplicates(s):
    result = []
    read = 0
    while read < len(s):
        c = s[read]
        if not result or c != result[-1]:
            result.append(c)
            read += 1
        else:
            while read < len(s) and s[read] == result[-1]:
                read += 1
            result.pop()
    return "".join(result)
'''


"""
This work: you really have to do it recursively
"""


def remove_duplicates(s):
    if not s:
        return s

    result = []
    prev = s[0]
    count = 1
    for c in s[1:]:
        if c == prev:
            count += 1
        else:
            if count == 1:
                result.append(prev)
            prev = c
            count = 1

    if count == 1:
        result.append(prev)
    return "".join(result)


def remove_adjacent_duplicates(s):
    result = remove_duplicates(s)
    while result != s:
        s = result
        result = remove_duplicates(s)
    return result


print(remove_adjacent_duplicates("mississipie"))
