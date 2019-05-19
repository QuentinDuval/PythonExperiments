class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    @classmethod
    def from_list(cls, xs):
        if type(xs) is not list:
            return cls(xs)
        elif not xs:
            return None
        else:
            node = cls(xs[0])
            node.left = cls.from_list(xs[1]) if len(xs) >= 2 else None
            node.right = cls.from_list(xs[2]) if len(xs) >= 3 else None
            return node

    @classmethod
    def depth(cls, node):
        max_depth = 0
        to_visit = [(node, 1)] if node else None
        while to_visit:
            node, depth = to_visit.pop()
            max_depth = max(depth, max_depth)
            if node.right:
                to_visit.append((node.right, depth + 1))
            if node.left:
                to_visit.append((node.left, depth + 1))
        return max_depth

    @classmethod
    def layout(cls, root):
        """
        A layout like this:

              1
            2     3
          4     6     7
        5           8

        Always shift one compared to the element below
        """
        max_depth = cls.depth(root)
        lines = ["" for _ in range(max_depth)]

        def visit(node, depth, shift):
            if node.left:
                shift = visit(node.left, depth + 1, shift)

            lines[depth] += (shift - len(lines[depth])) * " " + str(node.val)
            shift = len(lines[depth]) + 1
            if node.right:
                shift = visit(node.right, depth + 1, shift)
            return shift

        visit(root, depth=0, shift=0)
        return "\n".join(lines)
