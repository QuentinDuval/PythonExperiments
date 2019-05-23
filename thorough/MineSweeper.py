"""
https://leetcode.com/problems/minesweeper/

Let's play the minesweeper game (Wikipedia, online game)!

You are given a 2D char matrix representing the game board.
- 'M' represents an unrevealed mine
- 'E' represents an unrevealed empty square
- 'B' represents a revealed blank square that has no adjacent (above, below, left, right, and all 4 diagonals) mines
- digit ('1' to '8') represents how many mines are adjacent to this revealed square
- 'X' represents a revealed mine

Now given the next click position (row and column indices) among all the unrevealed squares ('M' or 'E'),
return the board after revealing this position according to the following rules:
- If a mine ('M') is revealed, then the game is over - change it to 'X'.
- If an empty square ('E') with no adjacent mines is revealed, then change it to revealed blank ('B') and all of its adjacent unrevealed squares should be revealed recursively.
- If an empty square ('E') with at least one adjacent mine is revealed, then change it to a digit ('1' to '8') representing the number of adjacent mines.

Return the board when no more squares will be revealed.
"""

from typing import List


class Solution:
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        if not board or not board[0]:
            return board

        height = len(board)
        width = len(board[0])

        def is_mine(x, y):
            return board[x][y] in {'M', 'X'}

        x, y = click
        if is_mine(x, y):
            board[x][y] = 'X'
            return board

        def adjacent_cells(i, j):
            for x in [i - 1, i, i + 1]:
                for y in [j - 1, j, j + 1]:
                    if 0 <= x < height and 0 <= y < width:
                        if (x, y) != (i, j):
                            yield (x, y)

        def count_mines_around(i, j):
            return sum(1 if is_mine(x, y) else 0 for x, y in adjacent_cells(i, j))

        to_visit = [(x, y)]
        discovered = {(x, y)}
        while to_visit:
            x, y = to_visit.pop()
            if board[x][y] != 'E':
                continue

            mine_count = count_mines_around(x, y)
            if mine_count > 0:
                board[x][y] = str(mine_count)
            else:
                board[x][y] = 'B'
                for i, j in adjacent_cells(x, y):
                    if (i, j) not in discovered:
                        discovered.add((i, j))
                        to_visit.append((i, j))

        return board
