"""
https://leetcode.com/problems/sum-of-subarray-minimums
"""


# TODO - can be done based on "next greater element"

"""
class Solution {
public:
    static const int MOD = 1000000007;
    
    int sumSubarrayMins(vector<int>& nums)
    {
        vector<pair<int, int>> val_to_index;
        for (int i = 0; i < nums.size(); ++i)
            val_to_index.push_back({nums[i], i});
        sort(val_to_index.begin(), val_to_index.end());
        
        set<int> boundaries;
        boundaries.insert(-1);
        boundaries.insert(nums.size());
        
        long long combinations = 0;
        for (auto [val, index]: val_to_index)
        {
            auto hi = boundaries.lower_bound(index);
            auto lo = prev(hi);
            long long count = ((index - *lo) * (*hi - index)) % MOD;
            combinations = (combinations + (count * val) % MOD) % MOD;
            boundaries.insert(index);
        }
        return combinations;
    }
};
"""
