class DisjointSets:
    """
    Data structure to implement Union-Find
    """

    def __init__(self, values):
        self.parents = list(range(len(values)))
        self.value_to_set = {v: i for i, v in enumerate(values)}

    def union(self, u, v):
        su = self.find(u)
        sv = self.find(v)
        if su != sv:
            self.parents[su] = sv   # TODO - Union by rank

    def find(self, u):
        s = self.value_to_set[u]
        while self.parents[self.parents[s]] != self.parents[s]:
            self.parents[s] = self.parents[self.parents[s]]
        return self.parents[s]

    def joined(self, u, v):
        return self.find(u) == self.find(v)

    def __repr__(self):
        return 'DisjointSets' + repr({
            'parents': self.parents,
            'value_to_set': self.value_to_set
        })
