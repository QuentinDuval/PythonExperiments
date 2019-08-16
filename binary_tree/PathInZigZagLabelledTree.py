"""
https://leetcode.com/problems/path-in-zigzag-labelled-binary-tree/

In an infinite binary tree where every node has two children, the nodes are labelled in row order.

In the odd numbered rows (ie., the first, third, fifth,...), the labelling is left to right, while in the
even numbered rows (second, fourth, sixth,...), the labelling is right to left.

Given the label of a node in this tree, return the labels in the path from the root of the tree to the
node with that label.
"""


from typing import List


class Solution:
    def pathInZigZagTree(self, n: int) -> List[int]:
        """
        This one is not easy to have exactly correct (really need to try on examples).

        First you have to compute where is the initial label in the tree:
        - compute the max power of 2 to find the stage (name it Q)
        - compute the remainder to find the offset (name it R)
        - correct the offset R if Q is odd (because of the counting)

        The we use the natural formula to go up the tree:
        - we systematically go up by dividing R by 2
        - then we find the offset by taking into account if Q is even or odd
        - we continue until q is positive

        Example for 26:
        - initially q is 4 (2 ** 4 = 16) and r is 26 - 16 = 10
        - then r = 5 and q = 3, leading to offset 2 ** q - 1 - 5 = 2 => 10
        - then r = 2 and q = 2, leading to offset 2 => 6
        - then r = 1 and q = 1, leading to offset 2 ** q - 1 - 1 = 0 => 2
        - then r = 0 and q = 0, leading to offset 0 => 1
        => result is [1, 2, 6, 10, 26]

        Complexity is O(log N)
        Beats 75%
        """

        path = [n]
        q = math.floor(math.log(n, 2))
        r = n % (2 ** q)
        if q % 2 == 1:
            r = 2 ** q - 1 - r

        while q > 0:
            r = r // 2
            q -= 1
            offset = 2 ** q - 1 - r if q % 2 == 1 else r
            path.append(2 ** q + offset)

        return path[::-1]
