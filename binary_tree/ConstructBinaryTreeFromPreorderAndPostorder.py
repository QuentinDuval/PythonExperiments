"""
https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/

Return any binary tree that matches the given preorder and postorder traversals.

Values in the traversals pre and post are distinct positive integers.
"""

from typing import List


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def constructFromPrePost(self, pre: List[int], post: List[int]) -> TreeNode:
        """
        First element of pre-order identifies the left tree
        Last element of post-order identifies the right tree
        When both are equal => several possible trees...

                  l     r
                v v     v
        pre  = [1,2,4,5,3,6,7]
        post = [4,5,2,6,7,3,1]
                    ^     ^ ^
                    l     r
        """

        def recur(pre, post):
            if not pre or not post:
                return None

            if len(pre) == 1:
                return TreeNode(pre[0])

            node = TreeNode(pre[0])
            l, r = pre[1], post[-2]
            if l == r:
                node.left = recur(pre[1:], post[:-1])
            else:
                start_r = pre.index(r)
                start_l = post.index(l)
                node.left = recur(pre[1:start_r], post[:start_l + 1])
                node.right = recur(pre[start_r:], post[start_l + 1:-1])
            return node

        return recur(pre, post)
