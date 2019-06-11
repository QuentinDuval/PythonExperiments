"""
https://leetcode.com/problems/zuma-game

Think about Zuma Game. You have a row of balls on the table, colored red(R), yellow(Y), blue(B), green(G), and white(W).
You also have several balls in your hand.

Each time, you may choose a ball in your hand, and insert it into the row (including the leftmost place and rightmost place).
Then, if there is a group of 3 or more balls in the same color touching, remove these balls.
Keep doing this until no more balls can be removed.

Find the minimal balls you have to insert to remove all the balls on the table. If you cannot remove all the balls, output -1.
"""

from collections import Counter


class Solution:
    def findMinStep(self, board: str, hand: str) -> int:
        """
        Given the constraints:
        - len(board) <= 20
        - len(hand) <= 5
        A brute force approach might work.

        We can at compress the input.
        RBYYBBRRB -> R1 B1 Y2 B2 R2 B

        We can compress the hand:
        GGYYR -> G2 Y2 R (and pack it into a small structure?)

        It seems there are OVERLAPPING SUB-PROBLEMS in the exploration:
        GYYRRG -> GRRG -> GG
        GYYRRG -> GYYG -> GG

        We might try memoization like like: https://leetcode.com/problems/remove-boxes
        But this is not easy:
        - how do you keep what you have left if you make it sequential?
        - how do you express dependence (sub-solutions depend on what is consumed)?
        => Seems DOOMED

        The solution below beat 99% without using any memoization.

        Note that the TRICK that makes it work is that we do not explore all possibilities:
        - we do not consider adding balls of a color where there is not a ball of that color already
        - we do not consider adding a single ball to a single ball of one color (we add 2 balls directly)
        => Still correct and reduces possibilities by a lot
        """
        if not board:
            return 0

        def zip_board():
            zipped = []
            prev_color = board[0]
            prev_count = 1
            for color in board[1:]:
                if prev_color == color:
                    prev_count += 1
                else:
                    zipped.append((prev_color, prev_count))
                    prev_color = color
                    prev_count = 1
            zipped.append((prev_color, prev_count))
            return zipped

        def play(board, i):
            if i == 0:
                return board[1:]
            if i == len(board) - 1:
                return board[:-1]

            if board[i + 1][0] != board[i - 1][0]:
                return board[:i] + board[i + 1:]

            new_count = board[i + 1][1] + board[i - 1][1]
            new_board = board[:i - 1] + [(board[i - 1][0], new_count)] + board[i + 2:]
            if new_count >= 3:
                return play(new_board, i - 1)
            return new_board

        def visit(board, hand):
            if not board: return 0
            if not hand: return float('inf')

            best = float('inf')
            for i, (color, count) in enumerate(board):
                missing = 3 - count
                if hand[color] >= missing:
                    hand[color] -= missing
                    best = min(best, missing + visit(play(board, i), hand))
                    hand[color] += missing
            return best

        res = visit(zip_board(), Counter(hand))
        return -1 if res == float('inf') else int(res)
