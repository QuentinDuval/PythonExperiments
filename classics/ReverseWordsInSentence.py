from typing import List


def reverse(chars: List[str], lo: int, hi: int):
    while lo < hi:
        chars[lo], chars[hi] = chars[hi], chars[lo]
        lo += 1
        hi -= 1


def reverse_words(sentence: List[str]):
    reverse(sentence, 0, len(sentence) - 1)

    lo = 0
    for hi in range(len(sentence)):
        if sentence[hi] == " ":
            reverse(sentence, lo, hi - 1)
            lo = hi + 1
    reverse(sentence, lo, len(sentence) - 1) # Do not forget the last word!


def test_reverse_words(sentence: str) -> str:
    l = list(sentence)
    reverse_words(l)
    return "".join(l)


print(test_reverse_words("Do or do not, there is no try, said Yoda"))
