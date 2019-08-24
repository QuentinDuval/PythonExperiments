
class SparseVector:
    def __init__(self, *key_value_pairs):
        # TODO - use a sorted set... with lower_bound
        self.values = list(key_value_pairs)
        self.values.sort(key=lambda p: p[0])

    def __mul__(self, other):
        """
        Multiply two sparse vectors by doing repetitive PARTIAL binary search
        """
        result = 0
        lo1, lo2 = 0, 0
        hi1 = len(self.values) - 1
        hi2 = len(other.values) - 1

        while lo1 <= hi1:
            # Look for index at index of first vector into second vector (using binary search)
            i1 = self.values[lo1][0]
            lo2 = other.binary_search_index(lo2, i1)
            if lo2 > hi2:
                break

            # If they match multiply the entries
            i2 = other.values[lo2][0]
            if i1 == i2:
                result += self.values[lo1][1] * other.values[lo2][1]
                lo1 += 1
                lo2 += 1

            # If they do not match advance the index in the first vector (using binary search)
            else:
                lo1 = self.binary_search_index(lo1, i2)
        return result

    def binary_search_index(self, lo, searched_index):
        hi = len(self.values) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2   # Equivalent to (hi + lo) // 2 but resilient to overflows
            index_at_mid = self.values[mid][0]
            if index_at_mid == searched_index:
                return mid
            elif index_at_mid < searched_index:
                lo = mid + 1
            else:
                hi = mid - 1
        return lo


"""
Example of multiplication of sparse vectors
"""
v1 = SparseVector([10, 5], [12, 10], [35, 6], [50, 1])
v2 = SparseVector([9, 5], [15, 10], [35, 10], [49, 2])
print(v1 * v2)
