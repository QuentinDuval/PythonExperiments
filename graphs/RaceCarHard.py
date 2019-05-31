"""
https://leetcode.com/problems/race-car

Your car starts at position 0 and speed +1 on an infinite number line.  (Your car can go into negative positions.)

Your car drives automatically according to a sequence of instructions A (accelerate) and R (reverse).

When you get an instruction "A", your car does the following: position += speed, speed *= 2.

When you get an instruction "R", your car does the following: if your speed is positive then speed = -1 , otherwise speed = 1.  (Your position stays the same.)

For example, after commands "AAR", your car goes to positions 0->1->3->3, and your speed goes to 1->2->4->-1.

Now for some target position, say the length of the shortest sequence of instructions to get there.
"""


from collections import deque


class Solution:
    def racecar(self, target: int) -> int:
        """
        BFS from the starting position until we reach the target:

        Complexity: O(2 ^ Result) since we expand the search by 2 almost every time
        But since Result is some kind of log N, it is O(N ** K) - TODO: what is K?

        Beats 22% (2548 ms)
        """
        if target == 0:
            return 0

        to_visit = deque()
        to_visit.append((0, 1))
        discovered = {(0, 1)}

        steps = 0
        while to_visit:
            for _ in range(len(to_visit)):
                pos, speed = to_visit.popleft()

                accelerate = (pos + speed, speed * 2)
                if accelerate[0] == target:
                    return steps + 1

                if accelerate not in discovered:
                    to_visit.append(accelerate)
                    discovered.add(accelerate)

                reverse = (pos, -1 if speed > 0 else 1)
                if reverse not in discovered:
                    to_visit.append(reverse)
                    discovered.add(reverse)

            steps += 1

        return steps
