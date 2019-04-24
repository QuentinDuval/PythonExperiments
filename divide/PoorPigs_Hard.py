import math


class Solution:
    def poorPigs(self, buckets: int, minutesToDie: int, minutesToTest: int) -> int:
        """
        The idea is clearly to makes use of the ability of a pig to drink several buckets

        --------------------------------
        WRONG IDEA
        --------------------------------

        If you have only one round (minutesToDie == minutesToTest)
        => you need 'buckets' - 1 pigs

        If you have two rounds:
        BEWARE: only one of these pigs die (you can reuse the others)
        - You could send a pig drink half of the buckets: max(1, 'buckets' / 2 + 1)
        - You could send 2 pigs drink 1/3 of the buckets: max(2, 'buckets' / 3 + 1)
        => Need to find the right split to minimize the number of pigs

        If you have 3 rounds => recursion

            def visit(buckets, rounds):
                if rounds == 1:
                    return buckets - 1

                min_pig = float('inf')
                for test_pig in range(1, buckets):
                    division = test_pig + 1
                    sub_buckets = math.ceil(buckets / division)
                    sub_sol = visit(sub_buckets, rounds - 1)
                    min_pig = min(min_pig, max(test_pig, 1 + sub_sol))
                return min_pig

            return visit(buckets, minutesToTest // minutesToDie + 1)

        But this is wrong ! Even the base case is wrong... COORDINATE SYSTEM IS THE KEY
        - You can use only 3 pigs to search among 8 buckets with 1 round:
        - Put the buckets in a 2 * 2 * 2 tensor: each pig tests one coordinate

        -------------------------
        https://leetcode.com/problems/poor-pigs/discuss/94266/Another-explanation-and-solution
        -------------------------
        """

        '''
        pigs = 0
        while (minutesToTest / minutesToDie + 1) ** pigs < buckets:
            pigs += 1
        return pigs
        '''

        pigs = math.log(buckets, minutesToTest / minutesToDie + 1)
        return math.ceil(pigs)
