"""
https://www.geeksforgeeks.org/find-minimum-time-to-finish-all-jobs-with-given-constraints/

Given an array of jobs with different time requirements.
There are K identical assignees available and we are given how much time an assignee takes to do one unit of the job.

Find the minimum time to finish all jobs with following constraints:
* An assignee can be assigned only contiguous jobs. For example, an assignee cannot be assigned jobs 1 and 3, but not 2.
* Two assignees cannot share (or co-assigned) a job.
"""

import bisect
from functools import lru_cache
from typing import List, Tuple


def minimum_time_dp(jobs: List[int], assignees: int, time_by_job: int) -> int:
    """
    First idea is to use Dynamic Programming
    - Try systematically all split points
    - Recurse with one less assigned

    There will be O(N * assignees) overlapping problems.
    The complexity is O(N ** 2 * assignees)
    """

    @lru_cache(maxsize=None)
    def visit(start: int, assignees: int) -> Tuple[int, List[int]]:
        if start == len(jobs):
            return 0, []

        if assignees == 1:
            return time_by_job * sum(jobs[start:]), [len(jobs)-1]

        min_time = float('inf')
        best_sol = None
        prefix_sum = 0
        for i in range(start, len(jobs)):
            prefix_sum += jobs[i] * time_by_job
            if prefix_sum > min_time:
                break

            sub_solution = visit(i+1, assignees-1)
            if max(prefix_sum, sub_solution[0]) < min_time:
                min_time = max(prefix_sum, sub_solution[0])
                best_sol = [i] + sub_solution[1]

        return min_time, best_sol

    return visit(0, assignees)


def minimum_time_divide(jobs: List[int], assignees: int, time_by_job: int) -> int:
    """
    A completely different approach is to do a binary search on the minimum time:
    - start with minimum time = total time
    - then try to reduce by 2 and check if it is possible
    - if possible, lower half recursion, else higher half recursion

    To check if something is possible:
    - do repeated binary searched (requires computing the prefix sums)
    - check if the last binary search reach past the end
    """

    '''
    # Linear check => leads to O(N log N) algorithm
    def is_possible(max_job_size: int) -> bool:
        partitions = 1
        cumulative_sum = 0
        for job in jobs:
            if cumulative_sum + job * time_by_job > max_job_size:
                partitions += 1
                if partitions > assignees:
                    return False
                cumulative_sum = job * time_by_job
            else:
                cumulative_sum += job * time_by_job
        return True
    '''

    # Log N check of feasibility
    prefix_sums = [0]
    for job in jobs:
        prefix_sums.append(prefix_sums[-1] + job * time_by_job)

    def is_possible(max_job_size: int) -> bool:
        partitions = 0
        current_pos = 1
        while current_pos < len(prefix_sums):
            partitions += 1
            if partitions > assignees:
                return False

            next_limit = prefix_sums[current_pos-1] + max_job_size
            current_pos = bisect.bisect_left(prefix_sums, next_limit + 1, lo=current_pos + 1)
        return True

    res = 0
    lo = 0
    hi = sum(jobs) * time_by_job
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if is_possible(mid):
            res = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return res


print(minimum_time_dp(assignees=4, time_by_job=5, jobs=[10, 7, 8, 12, 6, 8]))
print(minimum_time_divide(assignees=4, time_by_job=5, jobs=[10, 7, 8, 12, 6, 8]))
