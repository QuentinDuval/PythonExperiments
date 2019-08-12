"""
https://leetcode.com/problems/freedom-trail

In the video game Fallout 4, the quest "Road to Freedom" requires players to reach a metal dial called the
"Freedom Trail Ring", and use the dial to spell a specific keyword in order to open the door.

Given a string ring, which represents the code engraved on the outer ring and another string key, which represents
the keyword needs to be spelled. You need to find the minimum number of steps in order to spell all the characters
in the keyword.

Initially, the first character of the ring is aligned at 12:00 direction. You need to spell all the characters
in the string key one by one by rotating the ring clockwise or anticlockwise to make each character of the string
key aligned at 12:00 direction and then by pressing the center button.

At the stage of rotating the ring to spell the key character key[i]:

* You can rotate the ring clockwise or anticlockwise one place, which counts as 1 step.
  The final purpose of the rotation is to align one of the string ring's characters at the 12:00 direction,
  where this character must equal to the character key[i].

* If the character key[i] has been aligned at the 12:00 direction, you need to press the center button to spell,
  which also counts as 1 step. After the pressing, you could begin to spell the next character in the key (next stage),
  otherwise, you've finished all the spelling.
"""

from functools import lru_cache


class Solution:
    def findRotateSteps(self, ring: str, key: str) -> int:
        """
        There might be several letters on the ring, and so selecting the next position on the ring is not trivial.

        The idea is to explore the solutions and take the optimal one:
        - identify all the places where a letter is situated
        - compute the cost to get to this place
        - and recurse with a shorted key and a new starting position

        Dynamic programming is needed to avoid waste (overlapping sub-problems):
        - number of sub-problems is O(len(ring) * len(key))
        - cost by sub-problems is O(len(ring)) at worse
        """

        ring_pos_by_char = self.index_ring(ring)

        @lru_cache(maxsize=None)
        def visit(ring_pos: int, key_pos: int) -> int:
            if key_pos == len(key):
                return 0

            # Always better just to use the letter you are in
            if key[key_pos] == ring[ring_pos]:
                return 1 + visit(ring_pos, key_pos + 1)

            # Otherwise try to move at a position that has the letter, and continue from here
            min_cost = float('inf')
            letter = ord(key[key_pos]) - ord('a')
            for next_ring_pos in ring_pos_by_char[letter]:
                cost = self.cost_between(ring_pos, next_ring_pos, len(ring))
                min_cost = min(min_cost, 1 + cost + visit(next_ring_pos, key_pos + 1))
            return min_cost

        return visit(0, 0)

    def cost_between(self, source: int, destination: int, nb_slots: int) -> int:
        return min(
            abs(destination - source),
            abs((destination - nb_slots) - source),
            abs(destination - (source - nb_slots))
        )

    def index_ring(self, ring: str):
        index = [[] for _ in range(26)]
        for pos, c in enumerate(ring):
            index[ord(c) - ord('a')].append(pos)
        return index


