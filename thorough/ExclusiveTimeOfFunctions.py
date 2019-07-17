"""
https://leetcode.com/problems/exclusive-time-of-functions/

On a single threaded CPU, we execute some functions.  Each function has a unique id between 0 and N-1.

We store logs in timestamp order that describe when a function is entered or exited.

Each log is a string with this format: "{function_id}:{"start" | "end"}:{timestamp}".  For example, "0:start:3" means the function with id 0 started at the beginning of timestamp 3.  "1:end:2" means the function with id 1 ended at the end of timestamp 2.

A function's exclusive time is the number of units of time spent in this function.  Note that this does not include any recursive calls to child functions.

The CPU is single threaded which means that only one function is being executed at a given time unit.

Return the exclusive time of each function, sorted by their function id.
"""


from typing import List


class Solution:
    def exclusiveTime(self, n: int, logs: List[str]) -> List[int]:
        exclusive_time = [0] * n

        last_time = 0
        last_id = []

        for log in logs:
            fid, is_start, ts = self.parseLog(log)
            if is_start:
                if last_id:
                    exclusive_time[last_id[-1]] += ts - last_time
                last_id.append(fid)
                last_time = ts
            else:
                exclusive_time[fid] += ts + 1 - last_time
                last_id.pop()
                last_time = ts + 1

        return exclusive_time

    def parseLog(self, log):
        fid, event, ts = log.split(":")
        return int(fid), "start" == event, int(ts)
