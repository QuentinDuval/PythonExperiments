"""
Binary heap which supports a update-key
"""


class IndexHeap:
    def __init__(self):
        self.index = {}
        self.values = [(None, -1 * float('inf'))]

    def __len__(self):
        return len(self.values) - 1

    def __repr__(self):
        return 'IndexHeap' + repr({
            'index': self.index,
            'values': self.values
        })

    def min(self):
        return self.values[1][0]

    def pop_min(self):
        min_key, min_prio = self.values[1]
        if len(self.values) > 2:
            key, prio = self.values.pop()
            self.values[1] = (key, prio)
            self.index[key] = 1
            self._dive(1)
        else:
            self.values.pop()
        del self.index[min_key]
        return min_key, min_prio

    def add(self, key, priority):
        self.values.append((key, priority))
        last_index = len(self.values) - 1
        self.index[key] = last_index
        self._swim(last_index)

    def update(self, key, priority):
        idx = self.index[key]
        _, prev_priority = self.values[idx]
        if prev_priority != priority:
            self.values[idx] = (key, priority)
            if priority > prev_priority:
                self._dive(idx)
            else:
                self._swim(idx)

    def __contains__(self, key):
        return key in self.index

    def get_priority(self, key):
        return self.values[self.index[key]][1]

    def _swim(self, i):
        while self.values[i][1] < self.values[i//2][1]:
            self._swap(i, i//2)
            i = i // 2

    def _dive(self, i):
        while True:
            prio = self.values[i][1]
            l_prio = self.values[i*2][1] if i*2 < len(self.values) else float('inf')
            r_prio = self.values[i*2+1][1] if i*2+1 < len(self.values) else float('inf')
            if prio <= min(l_prio, r_prio):
                print(prio, l_prio, r_prio)
                break

            child = i*2 if l_prio < r_prio else i*2+1
            self._swap(i, child)
            i = child

    def _swap(self, i, j):
        ki = self.values[i][0]
        kj = self.values[j][0]
        self.index[ki], self.index[kj] = self.index[kj], self.index[ki]
        self.values[i], self.values[j] = self.values[j], self.values[i]
