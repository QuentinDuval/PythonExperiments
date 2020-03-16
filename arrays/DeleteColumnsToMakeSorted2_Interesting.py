"""
https://leetcode.com/problems/delete-columns-to-make-sorted-ii/

We are given an array A of N lowercase letter strings, all of the same length.

Now, we may choose any set of deletion indices, and for each string, we delete all the characters in those indices.

For example, if we have an array A = ["abcdef","uvwxyz"] and deletion indices {0, 2, 3}, then the final array after
deletions is ["bef","vyz"].

Suppose we chose a set of deletion indices D such that after deletions, the final array has its elements in
lexicographic order (A[0] <= A[1] <= A[2] ... <= A[A.length - 1]).

Return the minimum possible value of D.length.
"""


from typing import List


class Solution:
    def minDeletionSize(self, rows: List[str]) -> int:
        """
        If the first row is NOT SORTED: you need to remove it.
        If the first row is SORTED: you might still need to remove it (check recursively?).
        Tempted to split in sub-problems, but the sub-solutions might not be compatible...

        GREAT EXAMPLE:
            ["aaca"
            ,"abbb"
            ,"bbac"
            ,"babc"
            ,"cbcb"
            ,"dada"]
        Removal of 2nd column (because of row 2 and 3) => Removal of 3rd column.

        IDEA:
        - scan each column one by one, starting from the front
        - when you see an inversion, look at the last valid column: see if it is hidden
        """

        if not rows or not rows[0]:
            return 0

        h = len(rows)
        w = len(rows[0])

        selected_cols = []
        for col in range(w):
            removed = False
            for row in range(1, h):
                if rows[row][col] < rows[row - 1][col]:
                    removed = True
                    for selected in selected_cols:
                        if rows[row - 1][selected] < rows[row][selected]:
                            removed = False
                            break
                    if removed:
                        break
            if not removed:
                selected_cols.append(col)
        return w - len(selected_cols)
