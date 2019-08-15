"""
Sample question at Facebook:
https://www.facebook.com/watch/?v=10152735777427200
"""


from typing import List, Set


# Clarify the problem
# 1. Ask questions to the interviewer: all permutation of the letter, cost of the functions given as parameter
# 2. Work out some examples: try to find hedge cases (multiple time same input, etc).

# Design a solution
# 1. Think out loud when you design a solution
# 2. Explain your solution to your interviewer (clearly)
# 3. Wait for the interviewer to tell you to start coding (or ask)
# 4. DONE IS BETTER THAN PERFECT (the interview is not that long)

# Where you are done with your code:
# 1. Run against an example
# 2. Fix your bugs carefully
# 3. Indicate when you are done & sure it works
# 4. Fix your bugs carefully if reported by interviewer

# Your interview will ask if you can improve your solution
# - Take the opportunity to say what would do that is too complex
# - For instance, removing some cases, using lazy iterator, etc.

# Then you may ask some questions to Facebook
# - Gives insight on what motivates you
# - Can ask about how is it to work at Facebook, typical tasks, etc.


def nearby_words(s: str, get_nearby_chars, is_word) -> Set[str]:

    def backtrack(prefix: List[str], pos: int):
        if pos == len(s):
            word = "".join(prefix)
            if is_word(word):
                yield word
            return

        for c in get_nearby_chars(s[pos]):
            prefix.append(c)
            yield from backtrack(prefix, pos+1)
            prefix.pop()

    return set(backtrack([], 0))

