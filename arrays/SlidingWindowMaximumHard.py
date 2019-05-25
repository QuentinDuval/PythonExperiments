"""
https://leetcode.com/problems/sliding-window-maximum/

Given an array nums, there is a sliding window of size k which is moving from the very left of the array to the very right.
You can only see the k numbers in the window. Each time the sliding window moves right by one position.

Return the max of each sliding window.

FOLLOW UP: can you do it in LINEAR TIME.
"""


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

