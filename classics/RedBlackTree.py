class RedBlackTreeNode:
    def __init__(self, key, val, is_red=False):
        self.is_red = is_red
        self.key = key
        self.val = val
        self.left = None
        self.right = None


def rotate_left(node: RedBlackTreeNode):
    """
       b
      / \
     a  d
       / \
      c  e

    =>

        d
       / \
      b  e
     / \
    a  c
    """
    r = node.right
    node.right = r.left
    r.left = node
    r.is_red, node.is_red = node.is_red, r.is_red
    return r


def rotate_right(node: RedBlackTreeNode):
    """
        d
       / \
      b  e
     / \
    a  c

    =>

       b
      / \
     a  d
       / \
      c  e
    """
    l = node.left
    node.left = l.right
    l.right = node
    l.is_red, node.is_red = node.is_red, l.is_red
    return l


class RedBlackTree:
    def __init__(self):
        self.size: int = 0
        self.root: RedBlackTreeNode = None

    def __len__(self):
        return self.size

    def insert(self, key, val):

        def balance(node: RedBlackTreeNode) -> RedBlackTreeNode:

            # Two links right after insertion of right link
            if node.right and node.right.is_red and node.left and node.left.is_red:
                node.left.is_red = False
                node.right.is_red = False
                node.is_red = True
                return node

            # A right link to the right should link left after rotate
            if node.right and node.right.is_red:
                node = rotate_left(node)

            # After rotation, or because of red link insertion on level lower: two consecutive red links
            if node.left and node.left.left and node.left.is_red and node.left.left.is_red:
                node = rotate_right(node)   # Rotate left to have left and right red links
                node.left.is_red = False
                node.right.is_red = False
                node.is_red = True

            return node

        def recur(node: RedBlackTreeNode) -> RedBlackTreeNode:
            if node is None:
                self.size += 1
                return RedBlackTreeNode(key=key, val=val, is_red=True)

            if node.key == key:
                node.val = val
                return node

            if node.key < key:
                node.right = recur(node.right)
            else:
                node.left = recur(node.left)
            return balance(node)

        self.root = recur(self.root)
        self.root.is_red = False

    def find(self, key):
        """
        Normal binary search tree search:
        - start from the root
        - go right if the key is higher than the node
        - go left otherwise
        """
        node = self.root
        while node is not None:
            if node.key == key:
                return node.val
            if key > node.key:
                node = node.right
            else:
                node = node.left
        return None

    def delete(self, key):
        pass

    def smallest(self):
        pass

    def highest(self):
        pass

    def lower_bound(self, key):
        pass

    def upper_bound(self, key):
        pass

    def depth(self):
        max_depth = 0
        to_visit = [(self.root, 1)]
        while to_visit:
            node, depth = to_visit.pop()
            if node is not None:
                max_depth = max(max_depth, depth)
                to_visit.append((node.right, depth + 1))
                to_visit.append((node.left, depth + 1))
        return max_depth

    def draw(self):
        lines = []

        def recur(node: RedBlackTreeNode, depth: int, offset: int) -> int:
            if not node:
                return offset + 1

            if len(lines) <= depth:
                lines.append("")

            offset = recur(node.left, depth+1, offset)
            missing = offset - len(lines[depth])
            lines[depth] += " " * missing
            lines[depth] += str(node.val)
            offset = len(lines[depth])
            return recur(node.right, depth+1, offset)

        recur(self.root, 0, 0)
        for line in lines:
            print(line)


alphabet = "abcdefghijklmnopqrstuvwxyz"

m = RedBlackTree()
for i, c in enumerate(alphabet):
    m.insert(i, c)

for i, c in enumerate(alphabet):
    if c != m.find(i):
        print("Expected", c, "but got", m.find(i))

print(m.depth())
m.draw()

