"""
https://leetcode.com/problems/task-scheduler

Given a char array representing tasks CPU need to do. It contains capital letters A to Z where different letters represent different tasks. Tasks could be done without original order. Each task could be done in one interval. For each interval, CPU could finish one task or just be idle.

However, there is a non-negative cooling interval n that means between two same tasks, there must be at least n intervals that CPU are doing different tasks or just be idle.

You need to return the least number of intervals the CPU will take to finish all the given tasks.
"""

from typing import List


class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        """
        Take the task with the max amount of occurrence
        - Multiply the number of occurrences - 1 with 'n' + 1
        - Add 1 for every task with same number of occurrences
        """
        groups = {}
        for task in tasks:
            groups[task] = groups.get(task, 0) + 1

        max_count = 0
        max_occur = 0
        for count in groups.values():
            if count > max_occur:
                max_occur = count
                max_count = 1
            elif count == max_occur:
                max_count += 1

        return max(len(tasks), (max_occur - 1) * (n + 1) + max_count)

