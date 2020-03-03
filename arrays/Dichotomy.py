import string
from typing import *


CommitId = str


def is_test_failing(commit: CommitId) -> bool:
    return ord('f') < ord(commit[0])


def find_guilty_commit(commits: List[CommitId]) -> Optional[CommitId]:
    lo = 0
    hi = len(commits) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if is_test_failing(commits[mid]):
            hi = mid - 1
        else:
            lo = mid + 1
    return commits[lo] if lo < len(commits) else None


g = find_guilty_commit(string.ascii_lowercase)
print(g)
