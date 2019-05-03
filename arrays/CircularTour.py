"""
https://practice.geeksforgeeks.org/problems/circular-tour/1

Suppose there is a circle. There are N petrol pumps on that circle. You will be given two sets of data.
1. The amount of petrol that every petrol pump has.
2. Distance from that petrol pump to the next petrol pump.

Your task is to complete the function tour() which returns an integer denoting the first point from where a truck will
be able to complete the circle (The truck will stop at each petrol pump and it has infinite capacity).

Note :  Assume for 1 litre petrol, the truck can go 1 unit of distance.
"""


"""
A valid start 'i' necessarily has a fuel_amount[i] > distance_to_next[i]
If the sum of fuel_amount is lower than distance_to_next, there is no way to complete the circle

Solution 1 (brute force):
- Try every starting point accumulate from this, and check if it even goes negative
- Return the first starting point that never goes negative
=> Complexity is O(N**2)

Solution 2:
- Try to do a circular tour at position 0
- If it fails at index i (goes negative), start again from i+1
  (there is no point in testing in between, you can see that by drawing the integral)
- If it fails with a start at the end, there is no way to make it
  (alternatively, we could do a pre-check: if the sum of diffs is < 0, no way to do it) 
"""


def circular_tour(fuel_amount, dist_to_next):
    start = 0
    diffs = [fuel - dist for fuel, dist in zip(fuel_amount, dist_to_next)]
    while start < len(diffs):
        total = 0
        for delta in range(len(diffs)):
            total += diffs[(start + delta) % len(diffs)]
            if total < 0:
                start = start + delta + 1
                break
        else:
            return start
    return -1


print(circular_tour(fuel_amount=[4, 6, 7, 4], dist_to_next=[6, 5, 3, 5]))

print(circular_tour(fuel_amount=[3, 4, 6, 7, 4],
                    dist_to_next=[4, 6, 5, 3, 5]))
