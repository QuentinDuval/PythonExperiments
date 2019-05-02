"""
https://practice.geeksforgeeks.org/problems/rotten-oranges/0

Given a matrix of dimension r*c where each cell in the matrix can have values 0, 1 or 2 which has the following meaning:
- 0 : Empty cell
- 1 : Cells have fresh oranges
- 2 : Cells have rotten oranges

So, we have to determine what is the minimum time required to rot all oranges.
A rotten orange at index [i,j] can rot other fresh orange at indexes [i-1,j], [i+1,j], [i,j-1], [i,j+1] in 1 unit time.
If it is impossible to rot every orange then simply return -1.
"""


from collections import deque


def min_time_to_rot_oranges(height, width, matrix):
    """
    We can use the rotten oranges at the initial step for a BFS visitation of the matrix
    - For each initial cell tagged as 'fresh', keep first iteration at each it becomes rotten
    - At the end, look at each initially 'fresh' orange and see if there is one not rotten, otherwise takes highest
    """
    rotten_at = {}

    def neighbors(i, j):
        if i > 0:
            yield (i-1, j)
        if i < height - 1:
            yield (i+1, j)
        if j > 0:
            yield (i, j-1)
        if j < width - 1:
            yield (i, j+1)

    def visit_from(i, j):
        visited = set()
        toVisit = deque()
        toVisit.append((i, j, 0))   # You cannot do it with a stack, even if you keep the time...
        while toVisit:
            i, j, time = toVisit.popleft()
            if (i, j) in visited:
                continue

            visited.add((i, j))
            rotten_at[(i, j)] = min(rotten_at.get((i, j), float('inf')), time)
            for i2, j2 in neighbors(i, j):
                # It is critical to only go through fresh oranges (otherwise it always visit everything)
                if matrix[i2][j2] == 1 and (i2, j2) not in visited:
                    toVisit.append((i2, j2, time + 1))

    fresh = []
    for i in range(height):
        for j in range(width):
            if matrix[i][j] == 1:
                fresh.append((i, j))
            elif matrix[i][j] == 2:
                visit_from(i, j)

    max_time = 0
    for i, j in fresh:
        time = rotten_at.get((i, j), None)
        if time is None:
            return -1
        max_time = max(max_time, time)
    return max_time


matrix = [
    [2, 1, 0, 2, 1],
    [1, 0, 1, 2, 1],
    [1, 0, 0, 2, 1]
]

print(min_time_to_rot_oranges(3, 5, matrix))

matrix = [
    [1, 1, 1, 1, 1],
    [0, 2, 1, 1, 1]
]

print(min_time_to_rot_oranges(2, 5, matrix))
