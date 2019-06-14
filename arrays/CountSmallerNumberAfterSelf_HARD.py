"""
https://leetcode.com/problems/count-of-smaller-numbers-after-self/

You are given an integer array nums and you have to return a new counts array.
The counts array has the property where counts[i] is the number of smaller elements to the right of nums[i].
"""


from typing import List


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


"""
Solution based on binary searching the values already seen on the right.
Surprisingly, it passes all the tests, although the complexity is O(N**2).

Note: counting the elements in a STL map would lead to O(N**2) as well due to std::distance(begin(), lower_bound).

#include <map>

class Solution {
public:
    vector<int> countSmaller(vector<int>& nums)
    {        
        vector<int> smaller(nums.size());
        vector<int> rights;
        for (int i = nums.size() - 1; i >= 0; --i)
        {            
            auto insertion_point = lower_bound(rights.begin(), rights.end(), nums[i]);
            smaller[i] = std::distance(rights.begin(), insertion_point);
            rights.insert(insertion_point, nums[i]);
        }
        return smaller;
    }
};
"""


def lower_bound(nums, val):
    lo = 0
    hi = len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] < val:     # nums[lo] will end at first nums[lo] >= val
            lo = mid + 1
        else:
            hi = mid - 1
    return lo


class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        found = []
        n = len(nums)
        counts = [0] * n
        for i in reversed(range(n)):
            index = lower_bound(found, nums[i])
            counts[i] = index
            found.insert(index, nums[i])
        return counts


"""
Solutions based on binary search tree or merge sort
"""


# TODO
# https://leetcode.com/problems/count-of-smaller-numbers-after-self/discuss/305794/BST-solution-and-merge-sort-solution-written-in-C%2B%2B
