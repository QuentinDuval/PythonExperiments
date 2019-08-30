# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def findDuplicateSubtrees(self, root: TreeNode) -> List[TreeNode]:
        """
        Encode the sub-tries you find, and put them in a hash map
        """

        found = {}
        duplicates = []

        def visit(node: TreeNode) -> str:
            if not node:
                return "."

            r = str(node.val) + visit(node.left) + visit(node.right)
            prev_count = found.get(r, 0)
            found[r] = prev_count + 1
            if prev_count == 1:
                duplicates.append(node)
            return r

        visit(root)
        return duplicates
