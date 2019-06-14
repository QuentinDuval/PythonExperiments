"""
https://leetcode.com/problems/count-of-smaller-numbers-after-self/

You are given an integer array nums and you have to return a new counts array.
The counts array has the property where counts[i] is the number of smaller elements to the right of nums[i].
"""

"""
#include <map>

class Solution {
public:
    vector<int> countSmaller(vector<int>& nums)
    {
        /*
        We need to count the number of smallest elements on the right.
        This is equivalent to:

            Find the smallest element on the right, and add 1 to its count, put 0 otherwise.

        Just populate a sorted map from the right to the left, and search for the lowest value.

        THIS DOES NOT WORK: [2,0,1] returns [1,0,0] instead of [2,0,0]
        WHY? Because the next smallest value might be very far...
        */

        vector<int> smaller(nums.size());
        map<int, int> rights;

        for (int i = nums.size() - 1; i >= 0; --i)
        {
            int count_right = 0;
            auto insertion_point = rights.lower_bound(nums[i]);
            if (insertion_point != rights.begin())
            {
                if (insertion_point->first > nums[i])
                    --insertion_point;
                count_right = insertion_point->second + 1;
            }
            smaller[i] = count_right;
            rights[nums[i]] = count_right;
        }
        return smaller;
    }
};
"""

# TODO
