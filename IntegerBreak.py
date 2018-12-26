class Solution:
    def integerBreak(self, n):
        """
        2 => 1 + 1 => 1
        3 => 2 + 1 => 2
        4 => 2 + 2 => 4
        5 => 2 + 3 => 6
        6 => 3 + 3 (better than 2 + 2 + 2) => 9
        7 => 3 + 4 => 12
        8 => 2 + 6 (recur) => 18
             3 + 5 (recur) => 18
             4 + 4 => 16
        9 => 2 + 7 => 24
             3 + 6 => 27
             4 + 5 => 24
        10 => 1 + 9 => 27
              2 + 8 => 36
              3 + 7 => 36
              4 + 6 => 36
              5 + 5 => 36
        11 => 2 + 9 => 54
              3 + 8 => 54
              4 + 7 => 48
              5 + 6 => 54
        12 => 2 + 10 => 72
              3 + 9 => 81
              4 + 7 => 72
        """

        memo = [0] * (n + 1)
        memo[0:5] = [0, 0, 1, 2, 4]
        for i in range(5, n+1):
            memo[i] = 3 * max(i - 3, memo[i - 3])
        return memo[n]

    def check(self, n):
        memo = [0] * (n + 1)
        memo[0:5] = [0, 0, 1, 2, 4]
        for i in range(5, n + 1):
            for j in range(1, i // 2 + 1):
                memo[i] = max(memo[i], max(i - j, memo[i-j]) * max(j, memo[j]))
        return memo[n]


s = Solution()
for i in range(2, 120):
    print("Value:", s.integerBreak(i))
    print("Check:", s.check(i))