"""
https://leetcode.com/problems/min-cost-climbing-stairs

On a staircase, the i-th step has some non-negative cost cost[i] assigned (0 indexed).

Once you pay the cost, you can either climb one or two steps. You need to find minimum cost to reach the top of the
floor, and you can either start from the step with index 0, or the step with index 1.
"""

"""
class Solution {
public:
    int minCostClimbingStairs(vector<int>& costs)
    {
        // This is a classic DP problem
        // - you can either go top down and memoize but this is a waste here
        // - just go from the end to the beginning and remember 2 numbers (like fibo)
        
        int N = costs.size();
        int cost = costs[N-2];
        int next_cost = costs[N-1];
        for (int i = N-3; i >= 0; i--)
        {
            int new_cost = costs[i] + min(cost, next_cost);
            next_cost = cost;
            cost = new_cost;
        }
        return min(cost, next_cost);
    }
};
"""
