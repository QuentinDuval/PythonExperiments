"""
https://leetcode.com/problems/number-of-squareful-arrays

Given an array A of non-negative integers, the array is squareful if for every pair of adjacent elements,
their sum is a perfect square.

Return the number of permutations of A that are squareful.
Two permutations A1 and A2 differ if and only if there is some index i such that A1[i] != A2[i].
"""


"""
class Solution {
public:
    int numSquarefulPerms(vector<int>& nums)
    {
        // Idea: generate all permutations, but with back-tracking to avoid doing shit
        // Avoid the repetitions with same number twice (SORT TO HELP?)
        // Timeout if you divide by the combinations (m1! m2! ... where mi are the counts of each)
        unordered_map<int, int> counts;
        for (auto num: nums)
            counts[num] += 1;
        int count = generate(counts, nums.size(), -1);
        return count;
    }
    
    int generate(unordered_map<int, int>& counts, int remaining, int last)
    {
        if (remaining == 0)
            return 1;
        
        int total = 0;
        for (auto [key, count]: counts)
        {
            if (count == 0 || key == last)
                continue;
            
            if (last >= 0 && !is_square(key + last))
                continue;
            
            int max_count = 1;
            if (is_square(key + key))
                max_count = count;
            for (int i = 1; i <= max_count; ++i)
            {
                counts[key] = count - i;
                total += generate(counts, remaining-i, key);
            }
            counts[key] = count;
        }
        return total;
    }
    
    bool is_square(int val) const
    {
        int s = int(sqrt(val));
        return s * s == val;
    }
};
"""
