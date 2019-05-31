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
import heapq



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


"""
IMPORTANT:
If you want to skip steps in a BFS, you can use a HEAP that will return
the element with the min number of moves from the start.

We can use this here to optimize our search and avoid adding to the queue
elements that we know are of no importance.
"""


def cmp(a, b):
    if a < b: return -1
    if a > b: return 1
    return 0


class Solution(object):
    def racecar(self, target):
        if target == 0:
            return 0

        queue = [(0, 0, 1)]
        discovered = {(0, 1)}

        def add(curr, speed, move):
            if (curr, speed) not in discovered:
                discovered.add((curr, speed))
                heapq.heappush(queue, (move, curr, speed))

        while queue:
            move, curr, speed = heapq.heappop(queue)
            if curr == target:
                return move

            k = cmp(curr, target)  # -1 if need to go right, 1 if need to get left

            # If wrong direction (speed negative while need right)
            # Keep going in the wrong direction but not too far (3 / 2 * target away from 0)
            # ABSOLUTELY NECESSARY... allows to try all possibilities in wrong direction
            if k * speed > 0:
                if abs(curr + speed) < 3 / 2 * target:
                    add(curr + speed, speed * 2, move + 1)

            # Let us now try in the right direction, but in this case, we avoid doing all steps
            move += speed * k > 0  # If we have to change direction, count a 'R'
            speed = -k  # Start at absolute speed 1 in good direction
            while cmp(curr, target) * k > 0:  # while 'target' still on same side of 'curr'
                move += 1  # Double the speed (galoping search)
                curr += speed
                speed *= 2
            add(curr, speed, move)  # Try one overshooting target
            add(curr - speed / 2, k, move)  # Try the one before target
