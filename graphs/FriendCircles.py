"""
https://leetcode.com/problems/friend-circles/

There are N students in a class. Some of them are friends, while some are not. Their friendship is transitive in nature.
For example, if A is a direct friend of B, and B is a direct friend of C, then A is an indirect friend of C.
And we defined a friend circle is a group of students who are direct or indirect friends.

Given a N*N matrix M representing the friend relationship between students in the class.
If M[i][j] = 1, then the ith and jth students are direct friends with each other, otherwise not.
And you have to output the total number of friend circles among all the students.
"""


from typing import List


class Solution:
    def findCircleNum(self, friends: List[List[int]]) -> int:
        """
        For all students, try a DFS and tag each found element as in the same group
        """
        n = len(friends)

        groups = 0
        discovered = set()
        for start_student in range(n):
            if start_student not in discovered:
                groups += 1
                to_visit = [start_student]
                discovered.add(start_student)
                while to_visit:
                    current_student = to_visit.pop()
                    for student in range(n):
                        if friends[current_student][student] and student not in discovered:
                            discovered.add(student)
                            to_visit.append(student)

        return groups


"""
Slow version (144ms vs 40ms above):
- using 'visited' instead of 'discovered' (IMPORTANT DIFFERENCE)
- and thus using much more memory (storing the same element to visit several times)
"""


class Solution:
    def findCircleNum(self, friends: List[List[int]]) -> int:
        """
        For all students, try a DFS and tag each found element as in the same group
        """
        n = len(friends)

        groups = 0
        visited = set()
        for start_student in range(n):
            if start_student not in visited:
                groups += 1
                to_visit = [start_student]
                while to_visit:
                    current_student = to_visit.pop()
                    visited.add(current_student)
                    for student in range(n):
                        if friends[current_student][student] and student not in visited:
                            to_visit.append(student)

        return groups
