"""
https://leetcode.com/problems/keys-and-rooms

There are N rooms and you start in room 0.  Each room has a distinct number in 0, 1, 2, ..., N-1,
and each room may have some keys to access the next room.

Formally, each room i has a list of keys rooms[i], and each key rooms[i][j] is an integer in [0, 1, ..., N-1]
where N = rooms.length.  A key rooms[i][j] = v opens the room with number v.

Initially, all the rooms start locked (except for room 0).

You can walk back and forth between rooms freely.

Return true if and only if you can enter every room.
"""

from typing import List


class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        """
        This is a connectivity problem:
        - start from room 0 and visit recursively from there
        - if a room is not touched (set of visited not is not full) then we return False
        Beats 96%
        """
        discovered = {0}
        to_visit = [0]
        while to_visit:
            curr_room = to_visit.pop()
            for room in rooms[curr_room]:
                if room not in discovered:
                    to_visit.append(room)
                    discovered.add(room)
        return len(discovered) == len(rooms)
