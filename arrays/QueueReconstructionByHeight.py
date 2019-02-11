"""
https://leetcode.com/problems/queue-reconstruction-by-height

Suppose you have a random list of people standing in a queue.

Each person is described by a pair of integers (h, k), where h is the height of the person and k is
the number of people in front of this person who have a height greater than or equal to h.

Write an algorithm to reconstruct the queue.
"""


from typing import List


def reconstructQueue1(people: List[List[int]]) -> List[List[int]]:
    """
    Two possible algorithms
    Algorithm 1: Order the elements by increasing 'k'
    Algorithm 2: Order the elements by decreasing heights 'h' then increasing 'k'

    Algorithm 1 works, but it not optimal.
    Algorithm 2 is both faster and simpler.
    """

    """
    Algorithm 1:
    - Order the people by increasing 'k' (or number of bigger people before them)
    - Insert these people one by one where they would have 'k+1' bigger people before them

    Example:
    [[5,0], [7,0]]
    [[5,0], [7,0], [6,1], [7,1]]
    [[5,0], [7,0], [5,2], [6,1], [7,1]]
    [[5,0], [7,0], [5,2], [6,1], [4,4], [7,1]]

    Why does it work? Is there a risk of violating the placement of [6,1] with next elements?
    If there was an element taller than [6,1], placed before it, it would have a lower rank.
    But all the elements of lower rank are already placed, so there is no risk.

    Complexity: O(n^2)
    """

    # Sort by increasing rank and then height
    people.sort(key = lambda p: (p[1], p[0]))

    result = []
    for height, rank in people:
        if rank == 0:
            result.append([height, rank])
        else:
            r = 0
            insertion_point = 0
            for i in range(len(result)):
                if result[i][0] >= height:
                    r += 1
                if r > rank:
                    insertion_point = i
                    break
            else:
                insertion_point = len(result)
            result.insert(insertion_point, [height, rank])

    return result


def reconstructQueue2(people: List[List[int]]) -> List[List[int]]:
    """
    Algorithm 2
    - Order the people by decreasing height 'h' and increasing rank 'k'
    - Insert these people one by one at position 'k'

    Example (ordering is [[7,0], [7,1], [6,1], [5,0], [5,2], [4,4]]):
    [[7,0], [7,1]]
    [[7,0], [6,1], [7,1]]
    [[5,0], [7,0], [6,1], [7,1]]
    [[5,0], [7,0], [5,2], [6,1], [7,1]]
    [[5,0], [7,0], [5,2], [6,1], [4,4], [7,1]]

    Why does it work?
    - Once an element is placed, its 'k' cannot be violated with smaller elements.
    - You have to insert in increasing rank 'k' so that 'k' keeps its meaning

    Complexity: O(n^2)
    """

    people.sort(key = lambda p: (-p[0], p[1]))

    result = []
    for height, rank in people:
        result.insert(rank, [height, rank])
    return result



