"""
https://leetcode.com/problems/sliding-window-maximum/

Given an array nums, there is a sliding window of size k which is moving from the very left of the array to the very right.
You can only see the k numbers in the window. Each time the sliding window moves right by one position.

Return the max of each sliding window.

FOLLOW UP: can you do it in LINEAR TIME.
"""


from collections import deque
from typing import List


"""
Solution in O(N log K)
- Have an ordered map contain the window
- Add elements by incrementing the associated counter
- Remove elements when their counter reaches 0
- Always return the highest key in the map
"""


'''
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k)
    {   
        vector<int> maximums;
        if (nums.size() < k || nums.empty() || k == 0)
            return maximums;
            
        std::map<int, int> window;
        for (int i = 0; i < k; ++i)
            window[nums[i]] += 1;
        
        maximums.push_back(window.rbegin()->first);
        
        for (int i = k; i < nums.size(); ++i)
        {            
            window[nums[i]] += 1;
            auto start = window.find(nums[i-k]);
            if (start->second == 1)
                window.erase(start);
            else
                start->second -= 1;
            maximums.push_back(window.rbegin()->first);
        }
        
        return maximums;
    }
};
'''


"""        
This problem can be solved in O(N log N) rather easily:
- Use an ordered map (max heap would not do it)
- Add keys with count
- Remove keys with count == 0 (when slides out)
- Always return the maximum of the map

But this would work for any traveral of the collection
(and not necessarily sliding window).

So we do not use all the problem.
The solution is likely to involve a queue (becaus of the traversal).

Somehow we would like a max-fifo-queue that gives us the maximum in O(1)
"""


# TODO - small article on using all the data of your problem (the contrary to abstraction)


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        maximums = []
        if not nums or k == 0 or len(nums) < k:
            return maximums

        window = MaxQueue()
        for i in range(k):
            window.push(nums[i])

        maximums.append(window.get_max())
        for i in range(k, len(nums)):
            window.pop()
            window.push(nums[i])
            maximums.append(window.get_max())
        return maximums


class MaxQueue:
    def __init__(self):
        self.queue = deque()
        self.maxs = deque()  # Holds (max value, count) pairs

    def push(self, val):
        self.queue.append(val)
        equal_nb = 1
        while self.maxs and val >= self.maxs[-1][0]:
            if self.maxs[-1][0] == val:
                equal_nb += self.maxs[-1][1]
            self.maxs.pop()
        self.maxs.append((val, equal_nb))

    def pop(self):
        val = self.queue.popleft()
        if val == self.maxs[0][0]:
            count = self.maxs[0][1]
            self.maxs.popleft()
            if count > 1:
                self.maxs.appendleft((val, count - 1))
        return val

    def get_max(self):
        return self.maxs[0][0]



