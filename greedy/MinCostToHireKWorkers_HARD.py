"""
https://leetcode.com/problems/minimum-cost-to-hire-k-workers/

There are N workers.  The i-th worker has a quality[i] and a minimum wage expectation wage[i].

Now we want to hire exactly K workers to form a paid group.
When hiring a group of K workers, we must pay them according to the following rules:

Every worker in the paid group should be paid in the ratio of their quality compared to other workers in the paid group.
Every worker in the paid group must be paid at least their minimum wage expectation.
Return the least amount of money needed to form a paid group satisfying the above conditions.
"""


from typing import Tuple, List
import heapq


class Solution:
    def mincostToHireWorkers(self, quality: List[int], wage: List[int], K: int) -> float:
        """
        One idea:
        - sort by quality (both wage and quality)
        - move from left to right and keep last wage selected
        => Problem with this strategy is that the relation is bi-directional (future wages influence past wages => no ordering!)
        """

        '''
        n = len(quality)
        if K == 0:
            return 0

        indexes = list(range(n))
        indexes.sort(key=lambda i: quality[i])
        quality = [quality[i] for i in indexes]
        wage = [wage[i] for i in indexes]

        def visit(pos: int, last_wage: int, remaining: int) -> int:
            if remaining == 0:
                return 0

            min_cost = float('inf')
            for i in range(pos+1, n - remaining + 1):
                factor = quality[i] / quality[pos]
                new_wage = max(wage[i], last_wage * factor) # Does not work, you need to send back the information...
                cost = new_wage + visit(i, new_wage, remaining - 1)
                min_cost = min(min_cost, cost)
            return min_cost

        return min(wage[i] + visit(i, wage[i], K-1) for i in range(n-K+1))
        '''

        """
        Can we keep the idea of ordering?
        - Left to right, but the recursion returns a initial price + total cost
        - We adjust it: if the initial price goes up by 10%, everything goes up by 10% (ratios)

        But it does not lead to the right result either...
        => How do we know what next to select (think for k = 1)

        Plus the complexity is too high for 10,000 elements
        """

        '''
        n = len(quality)
        if K == 0:
            return 0

        indexes = list(range(n))
        indexes.sort(key=lambda i: quality[i])
        quality = [quality[i] for i in indexes]
        wage = [wage[i] for i in indexes]

        def visit(pos: int, remaining: int) -> Tuple[int, int]:
            if remaining == 0:
                return 0, 0

            min_cost = float('inf')
            min_first_wage = float('inf')

            for i in range(pos+1, n - remaining + 1):
                rec_wage, rec_cost = visit(i, remaining - 1)

                factor = quality[i] / quality[pos]                
                min_next_wage = wage[pos] * factor

                pos_wage = wage[pos]
                if rec_wage < min_next_wage:
                    rec_cost = wage[pos] + rec_cost * min_next_wage / rec_wage
                elif rec_wage > min_next_wage:
                    rec_cost += wage[pos] * rec_wage / factor
                    pos_wage = wage[pos] * rec_wage / factor
                else:
                    rec_cost += wage[pos]

                if rec_cost < min_cost:
                    min_cost = rec_cost
                    min_first_wage = pos_wage # not sure it will lead to best result

            return min_first_wage, min_cost

        return min(visit(i, K-1) for i in range(n-K+1))
        '''

        """
        Other strategy based on greedy

        Observe the problem:
        - consider employees in order or lower wage[i] / quality[i] ratio, then increasing quality[i]
            => this allows you always know the max wage[i] / quality[i] ratio
            => this allows you to only keep the sum of quality (and multiply it by this ratio)
        - then you need to get rid of the highest quality employee in any group of more than K employees
          (cause he costs the more, and the ratio wage[i] / quality[i] is fixes)
        - keep the minimum over moving this window over the inputs
        """

        n = len(quality)
        if K == 0:
            return 0

        employees = [(wage[i] / quality[i], quality[i]) for i in range(n)]
        employees.sort()

        min_cost = float('inf')
        quality_sum = 0
        min_quality_queue = []
        for employee in employees:
            quality_sum += employee[1]
            heapq.heappush(min_quality_queue, -employee[1])

            if len(min_quality_queue) > K:
                minus_max_quality = heapq.heappop(min_quality_queue)
                quality_sum += minus_max_quality

            if len(min_quality_queue) == K:
                wage_quality_ratio = employee[0]
                min_cost = min(min_cost, quality_sum * wage_quality_ratio)

        return min_cost
