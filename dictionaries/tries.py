

class NodeTrie:
    def __init__(self):
        self.valid = False
        self.children = {}

    def add(self, s):
        if not s:
            self.valid = True
        else:
            self.children.setdefault(s[0], NodeTrie()).add(s[1:])

    def sub_trie_for(self, s):
        if not s:
            return self

        child = self.children.get(s[0])
        return child.sub_trie_for(s[1:]) if child is not None else None

    def is_prefix(self, s):
        node = self.sub_trie_for(s)
        return node is not None

    def is_member(self, s):
        node = self.sub_trie_for(s)
        return node.valid if node is not None else False
