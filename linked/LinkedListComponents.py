"""
https://leetcode.com/problems/linked-list-components/

We are given head, the head node of a linked list containing unique integer values.

We are also given the list G, a subset of the values in the linked list.

Return the number of connected components in G, where two values are connected if they appear consecutively in the linked list.
"""

from typing import List


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def numComponents(self, head: ListNode, G: List[int]) -> int:
        """
        The crazy solution is to go for a graph
        """

        if not head:
            return 0

        graph = {}
        for node in G:
            graph[node] = []

        prev = head
        curr = head.next
        while curr:
            if prev.val in graph and curr.val in graph:
                graph[prev.val].append(curr.val)
                graph[curr.val].append(prev.val)
            prev, curr = curr, curr.next

        def dfs_from(visited, node):
            to_visit = [node]
            while to_visit:
                node = to_visit.pop()
                for adj in graph[node]:
                    if adj not in visited:
                        visited.add(adj)
                        to_visit.append(adj)

        visited = set()
        connected_component_count = 0
        for node in G:
            if node not in visited:
                connected_component_count += 1
                visited.add(node)
                dfs_from(visited, node)
        return connected_component_count

    def numComponents(self, head: ListNode, G: List[int]) -> int:
        """
        The wise solution is just to check if a node is missing in G
        """

        connected_component_count = 0
        G = set(G)

        size = 0
        prev = None
        curr = head
        while curr:
            if curr.val not in G:
                if size > 0:
                    connected_component_count += 1
                    size = 0
            else:
                size += 1
            prev, curr = curr, curr.next

        if size > 0:
            connected_component_count += 1
        return connected_component_count


