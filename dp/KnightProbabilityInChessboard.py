"""
https://leetcode.com/problems/knight-probability-in-chessboard

On an NxN chessboard, a knight starts at the r-th row and c-th column and attempts to make exactly K moves.
The rows and columns are 0 indexed, so the top-left square is (0, 0), and the bottom-right square is (N-1, N-1).

A chess knight has 8 possible moves it can make, as illustrated below. Each move is two squares in a cardinal direction,
then one square in an orthogonal direction.

Each time the knight is to move, it chooses one of eight possible moves uniformly at random (even if the piece would
go off the chessboard) and moves there.

The knight continues moving until it has made exactly K moves or has moved off the chessboard. Return the probability
that the knight remains on the board after it has stopped moving.
"""


class Solution:
    def knightProbability(self, N: int, K: int, init_r: int, init_c: int) -> float:
        """
        If we go like crazy, trying to explore K depth branch, we face an explonential 8^K

        Instead, we can compute the probability to go out out at each position of the knight:
        => O(N ^ 2)

        Then do the same with 2 moves:
        => O(N ^ 2)

        And so we end up with O(K * N ^ 2)
        """

        def moves_from(r, c):
            return [(r - 2, c - 1)
                , (r - 2, c + 1)
                , (r + 2, c - 1)
                , (r + 2, c + 1)
                , (r - 1, c - 2)
                , (r + 1, c - 2)
                , (r - 1, c + 2)
                , (r + 1, c + 2)]

        proba = [[1] * N for _ in range(N)]

        def get_proba(r, c):
            if r < 0 or r >= N or c < 0 or c >= N:
                return 0
            return proba[r][c]

        for i in range(K):
            new_proba = [[0] * N for _ in range(N)]
            for r in range(N):
                for c in range(N):
                    p = 0
                    for next_r, next_c in moves_from(r, c):
                        p += get_proba(next_r, next_c) / 8
                    new_proba[r][c] = p
            proba = new_proba
        return get_proba(init_r, init_c)
