"""
https://leetcode.com/problems/push-dominoes/
"""


class Solution:
    def pushDominoes(self, dominoes: str) -> str:
        """
        Since N can be big, there is no time to run a simulation

        A domino will fall to the right if:
        - it is preceeded by a 'R' and only '.' afterwards
        - there is not '.' followed by a 'L' that is closer
        (and symetrically for the falling to the left)

        So we can do two passes (left and right) to pre-compute:
        - count the closest 'R' on the left (only followed by '.')
        - count the closest 'L' to the right (only preceeded by '.')
        => Then, we just check for each point its situation.
        """

        N = len(dominoes)
        dist_R = [-1] * N
        dist_L = [-1] * N

        count = -1
        for i in range(N):
            if dominoes[i] == 'R':
                count = 0
            elif dominoes[i] == 'L':
                count = -1
            elif count >= 0:
                count += 1
            dist_R[i] = count if count >= 0 else 2 * N

        count = -1
        for i in reversed(range(N)):
            if dominoes[i] == 'L':
                count = 0
            elif dominoes[i] == 'R':
                count = -1
            elif count >= 0:
                count += 1
            dist_L[i] = count if count >= 0 else 2 * N

        # print(dist_R)
        # print(dist_L)

        out = ["."] * N
        for i in range(N):
            if dist_R[i] < dist_L[i]:
                out[i] = "R"
            elif dist_R[i] > dist_L[i]:
                out[i] = "L"
        return "".join(out)
