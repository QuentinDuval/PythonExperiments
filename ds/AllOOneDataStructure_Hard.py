"""
https://leetcode.com/problems/all-oone-data-structure/

Implement a data structure supporting the following operations:

* Inc(Key) - Inserts a new key with value 1. Or increments an existing key by 1.
  Key is guaranteed to be a non-empty string.
* Dec(Key) - If Key's value is 1, remove it from the data structure. Otherwise decrements an existing key by 1.
  If the key does not exist, this function does nothing.
  Key is guaranteed to be a non-empty string.
* GetMaxKey() - Returns one of the keys with maximal value. If no element exists, return an empty string "".
* GetMinKey() - Returns one of the keys with minimal value. If no element exists, return an empty string "".

Challenge: Perform all these in O(1) time complexity.
"""


class Node:
    def __init__(self, count: int, key: str):
        self.count = count
        self.keys = {key}
        self.prev = None
        self.next = None


class AllOne:
    """
    The Inc, Dec and Max key operations could be easily done via a:
    * counts: Map[Key, Int]: to store the count of each key
    * keys:   Map[Int, Set[Key]]: to know how many keys have the same count
    * A register for the Max:
        * just increment it when no entries in 'keys' at 'Inc'
        * just decrement it when no more entries in 'keys' when 'Dec'

    The problem is the 'Min' operation: sequences of Inc(a,1), Dec(a,1) for a
    non-existing entry 'a' are O(N) => need for each count to know which is
    the next count (and previous count as well to maintain the list)
    => We need a linked list of counts which knows how many keys points to it
    """

    def __init__(self):
        self.head = None
        self.tail = None
        self.index = {}

    def inc(self, key: str) -> None:
        if key in self.index:
            self._inc_key(key)
        elif self.head is not None:
            self._add_key(key)
        else:
            self._init_with(key)

    def _init_with(self, key: str):
        self.head = Node(1, key)
        self.tail = self.head
        self.index[key] = self.head

    def _add_key(self, key: str):
        if self.head.count == 1:
            self.head.keys.add(key)
            self.index[key] = self.head
        else:
            node = Node(1, key)
            node.next = self.head
            self.head.prev = node
            self.head = node
            self.index[key] = self.head

    def _del_node(self, node):
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

    def _inc_key(self, key: str):
        node = self.index[key]

        # Try to re-use the next node if possible
        if node.next is not None and node.next.count == node.count + 1:
            node.next.keys.add(key)
            node.keys.remove(key)
            self.index[key] = node.next
            if len(node.keys) == 0:
                self._del_node(node)

        # Try to re-use the current node if possible
        elif len(node.keys) == 1:
            node.count += 1

        # Otherwise, insert a new node in between
        else:
            new_node = Node(node.count+1, key)
            new_node.prev = node
            new_node.next = node.next
            node.next = new_node
            if new_node.next is None:
                self.tail = new_node
            else:
                new_node.next.prev = new_node
            node.keys.remove(key)
            self.index[key] = new_node

        '''
        # TODO - Update the head
        if node is self.head and node.prev:
            self.head = node.prev
        '''

    def dec(self, key: str) -> None:
        node = self.index.get(key)
        if node is None:
            return

        # Case of removing of the key
        if node.count == 1:
            if len(node.keys) > 1:
                node.keys.remove(key)
            else:
                self.head = node.next
                if not self.head:
                    self.tail = None
            del self.index[key]

        # Try to re-use the previous node if possible
        elif node.prev and node.prev.count == node.count - 1:
            node.prev.keys.add(key)
            node.keys.remove(key)
            self.index[key] = node.prev
            if len(node.keys) == 0:
                self._del_node(node)

        # Try to re-use the current node if possible
        elif len(node.keys) == 1:
            node.count -= 1

        # Otherwise, insert a new node in between
        else:
            new_node = Node(node.count-1, key)
            new_node.next = node
            new_node.prev = node.prev
            node.prev = new_node
            if new_node.prev is None:
                self.head = new_node
            else:
                new_node.prev.next = new_node
            node.keys.remove(key)
            self.index[key] = new_node

    def getMaxKey(self) -> str:
        if self.tail is not None:
            return next(iter(self.tail.keys))
        return ""

    def getMinKey(self) -> str:
        if self.head is not None:
            return next(iter(self.head.keys))
        return ""

    def __repr__(self):
        out = []
        node = self.head
        while node:
            token = "{" + str(node.count) + ":" + str(node.keys) + "}"
            out.append(token)
            node = node.next
        return "List: [" + ",".join(out) + "]\nIndex:" + str(self.index)


def test_all_o_one():
    a = AllOne()

    a.inc("a")
    print(a)
    assert a.getMinKey() == "a"
    assert a.getMaxKey() == "a"

    a.inc("b")
    a.inc("a")
    print(a)
    assert a.getMinKey() == "b"
    assert a.getMaxKey() == "a"

    a.inc("b")
    a.inc("a")
    a.inc("a")
    a.inc("c")
    print(a)
    assert a.getMinKey() == "c"
    assert a.getMaxKey() == "a"

    a.inc("c")
    a.dec("c")
    print(a)
    assert a.getMinKey() == "c"
    assert a.getMaxKey() == "a"

    a.dec("c")
    print(a)
    assert a.getMinKey() == "b"
    assert a.getMaxKey() == "a"

    for _ in range(10):
        a.dec("a")
    print(a)
    assert a.getMinKey() == "b"
    assert a.getMaxKey() == "b"


if __name__ == '__main__':
    test_all_o_one()
