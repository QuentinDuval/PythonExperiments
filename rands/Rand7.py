

"""
Rand10 with Rand7
https://leetcode.com/problems/implement-rand10-using-rand7/
"""


def rand7():
    import random
    return random.randint(1, 7)


def rand10() -> int:
    def rand6():
        r = 7
        while r > 6:
            r = rand7()
        return r

    def rand5():
        r = 7
        while r > 5:
            r = rand7()
        return r

    if rand6() <= 3:
        return rand5()
    else:
        return rand5() + 5