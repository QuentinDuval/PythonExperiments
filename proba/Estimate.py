"""
Given:
    x = random.uniform(0, 2)
    y = random.uniform(-4, 0)
What is the probability of x + y being between -1 and 1
"""


"""
Solution:

PHASE 1: RECAST THE PROBLEM
y = -4 + random.uniform(0, 4)

Given:
    x = random.uniform(0, 2)
    y = random.uniform(0, 4)
What is the probability of x + y being between 3 and 5


PHASE 2: BASIC ANALYSIS
To get some transformations that will help us:

p (3 < x + y < 5) = p(x + y < 5) - p (x + y < 3)

If k is between 0 and 4:
p (y < k) = Integral of 1/4 dx over [0, k] = k / 4

So if 0 <= k1 < k2 <= 4:
p (k1 < y < k2) = (k2 - k1) / 4


PHASE 3: SOLVE THE PROBLEM (BY CUTTING IT)
To avoid having bad integrals (the cumulative expectations are of the form 0 -> k * x -> 1 with borns)

* IF x > 1 (p = 1/2):
    - x + y < 5 if y < 5 - x => p = (5 - x) / 4
    - x + y < 3 if y < 3 - x => p = (3 - x) / 4
    - 3 < x + y < 5 = subtract both = 1 / 2

* If x < 1 (p = 1/2):
    - x + y < 5 if y < 4 => p = 1 (all y will lead to this)
    - x + y < 3 if y < 3 - x => p = (3 - x) / 4
    - 3 < x + y < 5 = subtract both = (1 + x) / 4 = 3 / 8 on average

Total probability = 1 / 2 * 1 / 2 + 1 / 2 * 3 / 8 = 7 / 16
"""


"""
Experimentally
"""

import numpy as np


sample_count = 100000
xs = np.random.uniform(0, 2, sample_count)
ys = np.random.uniform(-4, 0, sample_count)

count = 0
match = 0
for x, y in zip(xs, ys):
    count += 1
    if -1 <= x + y <= 1:
        match += 1

print(7 / 16)
print(match / count)
