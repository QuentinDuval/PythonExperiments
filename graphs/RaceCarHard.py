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
        if target == 0:
            return 0

        to_visit = deque()
        to_visit.append((0, 1))
        discovered = {(0, 1)}

        def add(x):
            if x not in discovered:
                to_visit.append(x)
                discovered.add(x)

        steps = 0
        while to_visit:
            for _ in range(len(to_visit)):
                pos, speed = to_visit.popleft()

                accelerate = (pos + speed, speed * 2)
                if accelerate[0] == target:
                    return steps + 1

                add(accelerate)
                add((pos, -1 if speed > 0 else 1))

            steps += 1

        return steps
