import functools


"""
Put split points on a collection to share it as equally as possible
- The measure of equally is the LARGEST value (try to minimize it)
"""


def optimal_contiguous_partionning(quantities, k):
    n = len(quantities)

    @functools.lru_cache(maxsize=None)
    def visit(i, k):
        if k == 1:
            return sum(quantities[i:])
        if i == n:
            return 0

        min_max = float('inf')
        for j in range(i+1, n):
            this_partition = sum(quantities[i:j])
            min_max = min(min_max, max(this_partition, visit(j, k-1)))
        return min_max

    return visit(0, k)


print(optimal_contiguous_partionning([1, 2, 3, 4, 5, 6, 7, 8, 9], k=3))


