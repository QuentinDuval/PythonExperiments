"""
https://leetcode.com/problems/lru-cache
"""


class LRUCache:

    class Node:
        def __init__(self, key, val, prev=None, next=None):
            self.key = key
            self.val = val
            self.prev = prev
            self.next = next

    def __init__(self, max_size):
        self.head = None
        self.tail = None
        self.mapping = dict()
        self.max_size = max_size

    def put(self, key: int, value: int) -> None:
        node = self.mapping.get(key)
        if node is None:
            node = self.Node(key, value)
            self.mapping[key] = node
            self._add_front(node)
            self._check_overflow()
        else:
            node.val = value
            self._move_front(node)

    def get(self, key: int) -> int:        
        node = self.mapping.get(key)
        if node is not None:
            self._move_front(node)            
            return node.val
        return -1

    def _add_front(self, node):
        if self.head is not None:
            node.prev = None
            node.next = self.head
            self.head.prev = node
            self.head = node
        else:
            node.prev = None
            node.next = None
            self.head = node
            self.tail = node

    def _check_overflow(self):
        while len(self.mapping) > self.max_size:
            del self.mapping[self.tail.key]
            self.tail = self.tail.prev
            self.tail.next = None
            if self.tail is None:
                self.head = None

    def _move_front(self, node):        
        if node is self.head:
            return

        if node is self.tail:
            self.tail = node.prev
            self.tail.next = None
            self._add_front(node)
            return

        node.prev.next = node.next
        node.next.prev = node.prev
        self._add_front(node)