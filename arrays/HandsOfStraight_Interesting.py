"""
https://leetcode.com/problems/hand-of-straights

Alice has a hand of cards, given as an array of integers.

Now she wants to rearrange the cards into groups so that each group is size W, and consists of W consecutive cards.

Return true if and only if she can.
"""

from typing import List


class Solution:
    def isNStraightHand(self, hand: List[int], W: int) -> bool:
        """
        The idea is to have a list of straight hand in constructions.

        Sorting the hand and moving through the list in increasing order:
        - if we have a list in construction whose last card is 2 back or more, we failed
        - if we have a list in construction whose last card is 1 back, complete it
        - else we create a new list

        To find the right list to append, we can use a queue to go through the list in construction
        in turns, and not loop through all of them.
        """

        if W <= 1:
            return True

        stacks = deque()
        for card in sorted(hand):
            if stacks:
                s = stacks.popleft()
                if s[-1] == card:
                    stacks.append(s)
                    stacks.append([card])
                elif s[-1] + 1 < card:
                    return False
                elif len(s) < W - 1:
                    s.append(card)
                    stacks.append(s)
            else:
                stacks.append([card])
        return not stacks
