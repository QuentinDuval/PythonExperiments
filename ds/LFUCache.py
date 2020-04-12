"""
https://leetcode.com/problems/lfu-cache/
"""


class Node:
    def __init__(self, key: int, value: int):
        self.key = key
        self.value = value
        self.frequency = 1
        self.prev = None
        self.next = None

    def __repr__(self):
        return str({'key': self.key,
                    'value': self.value,
                    'freq': self.frequency,
                    'prev': self.prev.key if self.prev else None,
                    'next': self.next.key if self.next else None})


class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.key_index = {}     # points to the node having the correct key
        self.freq_index = {}    # points to the last node with given frequency
        self.head = None
        self.tail = None

    def get(self, key: int) -> int:
        if self.capacity == 0:
            return -1

        node = self.key_index.get(key)
        if node is None:
            return -1

        node.frequency += 1
        self._move_node(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if self.capacity == 0:
            return

        if self.capacity == 1:
            node = Node(key, value)
            self.head = node
            self.tail = node
            self.key_index = {key: node}
            self.freq_index = {1: node}
            return

        if not self.head:
            node = Node(key, value)
            self.head = node
            self.tail = node
            self.key_index[key] = node
            self.freq_index[1] = node
            return

        node = self.key_index.get(key)
        if node is None:
            self._evict_if_necessary()
            node = Node(key, value)
            node.next = self.head
            self.head.prev = node
            self.head = node
            self.key_index[key] = node
            self._move_node(node)
        else:
            node.value = value
            node.frequency += 1
            self._move_node(node)

    def _move_node(self, node):
        prev_node, next_node = self._detach_node(node)

        if not self.head:
            self.head = node
            self.tail = node
            return

        ins_node = self.freq_index.get(node.frequency)
        if not ins_node:
            if next_node and next_node.frequency > node.frequency:
                ins_node = next_node.prev
            else:
                ins_node = self.tail
        if ins_node:
            node.next = ins_node.next
            node.prev = ins_node
            if ins_node.next:
                ins_node.next.prev = node
            ins_node.next = node
            self.freq_index[node.frequency] = node
            if ins_node is self.tail:
                self.tail = node

        if not ins_node:
            self.head.prev = node
            node.next = self.head
            self.head = node
            self.freq_index[node.frequency] = node

    def _detach_node(self, node):
        prev_node = node.prev
        next_node = node.next
        node.prev = None
        node.tail = None

        if node is self.head:
            self.head = next_node
        else:
            prev_node.next = next_node

        if node is self.tail:
            self.tail = prev_node
        else:
            next_node.prev = prev_node

        if self.freq_index.get(node.frequency-1) is node:
            if prev_node and prev_node.frequency == node.frequency-1:
                self.freq_index[node.frequency-1] = prev_node
            else:
                del self.freq_index[node.frequency-1]

        return prev_node, next_node

    def _evict_if_necessary(self):
        if self.capacity > len(self.key_index):
            return

        node = self.head
        self.head = self.head.next
        if node is self.freq_index[node.frequency]:
            del self.freq_index[node.frequency]
        del self.key_index[node.key]

    def describe(self):
        print("-" * 10)
        node = self.head
        while node:
            print(node.key, node.value, node.frequency)
            node = node.next
        assert self.head
        assert self.tail


def test_lfu_1():
    cache = LFUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    cache.describe()
    assert cache.get(1) == 1
    cache.describe()
    cache.put(3, 3)
    cache.describe()
    assert cache.get(2) == -1   # Evicted
    assert cache.get(3) == 3
    cache.put(4, 4)
    cache.describe()
    assert cache.get(1) == -1   # Evicted
    cache.describe()
    assert cache.get(3) == 3
    cache.describe()
    assert cache.get(4) == 4


def test_lfu_2():
    cache = LFUCache(2)
    cache.put(3, 1)
    cache.describe()
    cache.put(2, 1)
    cache.describe()
    cache.put(2, 2)
    cache.describe()
    cache.put(4, 4)
    cache.describe()
    print(cache.get(2))
    assert cache.get(2) == 2


def test_lfu_3():
    cache = LFUCache(0)
    cache.put(0, 0)
    assert cache.get(0) == -1

    cache = LFUCache(1)
    cache.put(0, 0)
    assert cache.get(0) == 0


def test_lfu_4():
    cache = LFUCache(2)
    cache.put(2, 1)
    cache.put(2, 2)
    assert cache.get(2) == 2
    cache.put(1, 1)
    cache.put(4, 1)
    assert cache.get(2) == 2


def run_scenario(cache_size: int, ops, args, res):
    cache = LFUCache(cache_size)
    for o, a, r in zip(ops, args, res):
        print(o, a, "=>", r)
        if o == "put":
            cache.put(*a)
        elif o == "get":
            got = cache.get(*a)
            if r != got:
                print("got", got, "expected", r)
                assert r == got


def test_lfu_5():
    ops = ["put", "put", "put", "put", "put", "get", "put", "get", "get", "put", "get", "put", "put", "put",
         "get", "put", "get", "get", "get", "get", "put", "put", "get", "get", "get", "put", "put", "get", "put", "get",
         "put", "get", "get", "get", "put", "put", "put", "get", "put", "get", "get", "put", "put", "get", "put", "put",
         "put", "put", "get", "put", "put", "get", "put", "put", "get", "put", "put", "put", "put", "put", "get", "put",
         "put", "get", "put", "get", "get", "get", "put", "get", "get", "put", "put", "put", "put", "get", "put", "put",
         "put", "put", "get", "get", "get", "put", "put", "put", "get", "put", "put", "put", "get", "put", "put", "put",
         "get", "get", "get", "put", "put", "put", "put", "get", "put", "put", "put", "put", "put", "put", "put"]
    args = [[10, 13], [3, 17], [6, 11], [10, 5], [9, 10], [13], [2, 19], [2], [3], [5, 25], [8], [9, 22], [5, 5],
         [1, 30], [11], [9, 12], [7], [5], [8], [9], [4, 30], [9, 3], [9], [10], [10], [6, 14], [3, 1], [3], [10, 11], [8],
         [2, 14], [1], [5], [4], [11, 4], [12, 24], [5, 18], [13], [7, 23], [8], [12], [3, 27], [2, 12], [5], [2, 9],
         [13, 4], [8, 18], [1, 7], [6], [9, 29], [8, 21], [5], [6, 30], [1, 12], [10], [4, 15], [7, 22], [11, 26], [8, 17],
         [9, 29], [5], [3, 4], [11, 30], [12], [4, 29], [3], [9], [6], [3, 4], [1], [10], [3, 29], [10, 28], [1, 20],
         [11, 13], [3], [3, 12], [3, 8], [10, 9], [3, 26], [8], [7], [5], [13, 17], [2, 27], [11, 15], [12], [9, 19],
         [2, 15], [3, 16], [1], [12, 17], [9, 1], [6, 19], [4], [5], [5], [8, 1], [11, 7], [5, 2], [9, 28], [1], [2, 2],
         [7, 4], [4, 22], [7, 24], [9, 26], [13, 28], [11, 26]]
    res = [None,None,None,None,None,-1,None,19,17,None,-1,None,None,None,-1,None,-1,5,-1,12,None,None,3,5,5,None,
           None,1,None,-1,None,30,5,30,None,None,None,-1,None,-1,24,None,None,18,None,None,None,None,14,None,None,
           18,None,None,11,None,None,None,None,None,18,None,None,-1,None,4,29,30,None,12,11,None,None,None,None,29,
           None,None,None,None,17,-1,18,None,None,None,-1,None,None,None,20,None,None,None,29,18,18,None,None,None,
           None,20,None,None,None,None,None,None,None]
    run_scenario(10, ops, args, res)


if __name__ == '__main__':
    test_lfu_1()
    test_lfu_2()
    test_lfu_3()
    test_lfu_4()
    test_lfu_5()

