"""
https://leetcode.com/problems/dota2-senate/

In the world of Dota2, there are two parties: the Radiant and the Dire.

The Dota2 senate consists of senators coming from two parties. Now the senate wants to make a decision about a change
in the Dota2 game. The voting for this change is a round-based procedure.

In each round, each senator can exercise one of the two rights:

* Ban one senator's right:
  A senator can make another senator lose all his rights in this and all the following rounds.
* Announce the victory:
  If this senator found the senators who still have rights to vote are all from the same party, he can announce the
  victory and make the decision about the change in the game.

Given a string representing each senator's party belonging. The character 'R' and 'D' represent the Radiant party and
the Dire party respectively. Then if there are n senators, the size of the given string will be n.

The round-based procedure starts from the first senator to the last senator in the given order. This procedure will last
until the end of voting. All the senators who have lost their rights will be skipped during the procedure.

Suppose every senator is smart enough and will play the best strategy for his own party, you need to predict which party
will finally announce the victory and make the change in the Dota2 game. The output should be Radiant or Dire.
"""


class Solution:
    def predictPartyVictory(self, senate: str) -> str:
        """
        If there is a guy from another party after, you have to 'ban' and you cannot 'vote'
        It it always better to ban the next opponent (from left to right) so that you can ban more
        If you already banned all the ones after, ban from the start
        """

        while len(set(senate)) > 1:
            banned_r = 0
            banned_d = 0
            next_senate = ""

            # Start by banning all the guys on the right if you can, to limit their rights
            for c in senate:
                if c == 'R':
                    if banned_r:
                        banned_r -= 1
                    else:
                        banned_d += 1
                        next_senate += c
                elif c == 'D':
                    if banned_d:
                        banned_d -= 1
                    else:
                        banned_r += 1
                        next_senate += c

            # Then for the remaining bans, start from the left (limit the other in the round - like a circular array)
            senate = next_senate
            next_senate = ""
            for c in senate:
                if c == 'R':
                    if banned_r:
                        banned_r -= 1
                    else:
                        next_senate += c
                elif c == 'D':
                    if banned_d:
                        banned_d -= 1
                    else:
                        next_senate += c
            senate = next_senate

        if senate[0] == 'R':
            return "Radiant"
        else:
            return "Dire"
