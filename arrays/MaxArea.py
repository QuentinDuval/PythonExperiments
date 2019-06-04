from typing import List



def max_area(heights: List[int]) -> int:
    """
    Two fingers 'lo' and 'hi':

    if heights[lo] < heights[hi]:
        lo += 1
    else:
        hi -= 1

    Why does it work?
    - Clearly, we can only improve a solution by increasing the heights either on left or right
    - We need to increase the shortest height, otherwise we might miss a solution
    - If both heights are equal, it does not matter which we increase:
        if we increase left, then the max will not go up (since it is the minimum of the heights that count)
        so we will need to increase right anyway, and it will happens since we will have left > right

    Why does it work for real?
    When we have i < j and heights[i] < heights[j], it is clear that it is useless to try 'i' more to the right since the
    heights of j will be the limiting factor and the width will decrease.
    => So we basically eliminate all 'i2' > 'i"
    """

    # TODO
