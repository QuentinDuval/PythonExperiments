"""
https://leetcode.com/problems/single-number-iii/

Given an array of numbers nums, in which exactly two elements appear only once and all the other elements appear exactly
twice. Find the two elements that appear only once.
"""


"""
class Solution {
public:
    vector<int> singleNumber(vector<int> const& nums)
    {
        // XOR on the whole list
        // => gives you the bits that appears odd times
        // => gives you the bits different in both unique numbers
        int total_xor = 0;
        for (int num: nums)
            total_xor ^= num;

        // Second pass: XOR filtered on numbers with one of these bits set
        // => will give you one of these unique numbers
        int lowest_bit = 0;
        for (; lowest_bit < 32; ++lowest_bit)
            if ((1 << lowest_bit) & total_xor)
                break;

        int partial_xor = 0;
        for (int num: nums)
            if (num & (1 << lowest_bit))
                partial_xor ^= num;

        // Now you only need to subtract
        return {partial_xor, partial_xor ^ total_xor};
    }
};
"""
