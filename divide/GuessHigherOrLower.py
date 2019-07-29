"""
https://leetcode.com/problems/guess-number-higher-or-lower/

We are playing the Guess Game. The game is as follows:

I pick a number from 1 to n. You have to guess which number I picked.

Every time you guess wrong, I'll tell you whether the number is higher or lower.

You call a pre-defined API guess(int num) which returns 3 possible results (-1, 1, or 0):

-1 : My number is lower
 1 : My number is higher
 0 : Congrats! You got it!
"""

"""
// Forward declaration of guess API.
// @param num, your guess
// @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
int guess(int num);

class Solution {
public:
    int guessNumber(int n) {
        int lo = 1;
        int hi = n;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            int res = guess(mid);
            if (res == 0)
                return mid;

            if (res < 0) {
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }
};
"""
