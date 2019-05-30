"""
https://leetcode.com/problems/data-stream-as-disjoint-intervals/

Given a data stream input of non-negative integers a1, a2, ..., an, ...,
summarize the numbers seen so far as a list of disjoint intervals.

For example, suppose the integers from the data stream are 1, 3, 7, 2, 6, ..., then the summary will be:
[1, 1]
[1, 1], [3, 3]
[1, 1], [3, 3], [7, 7]
[1, 3], [7, 7]
[1, 3], [6, 7]
"""


'''
#include <map>

class SummaryRanges {
    std::map<int, int> starts;

public:
    SummaryRanges() {}

    void addNum(int val) {
        int start_interval = val;
        int end_interval = val;

        auto lower = starts.lower_bound(val);
        if (lower != starts.end()) {
            if (lower->first == val) // If already covered, exit
                return;
            if (lower->first == val + 1) // Try to extend scope
                end_interval = lower->second;
        }
        if (lower != starts.begin()) {
            auto prev = std::prev(lower);
            if (prev->second >= val) // If already covered, exit
                return;
            if (prev->second == val - 1) // Try to extend scope
                start_interval = prev->first;
        }

        if (end_interval != val) {
            starts.erase(lower);
        }

        starts[start_interval] = end_interval;
    }

    vector<vector<int>> getIntervals() const {
        vector<vector<int>> res;
        for (auto const& interval: starts)
            res.push_back({ interval.first, interval.second });
        return res;
    }
};
'''
