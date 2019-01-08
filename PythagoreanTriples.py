from collections import *
from functools import *
import itertools
import numpy as np
import timeit


Timed = namedtuple('Timed', ['result', 'elapsed'])


def timed(f):
    @wraps(f)
    def timed_f(*args, **kwargs):
        start = timeit.default_timer()
        res = f(*args, **kwargs)
        end = timeit.default_timer()
        return Timed(result=res, elapsed=end-start)
    return timed_f


def gcd(a, b):
    a = min(a, b)
    b = max(a, b)
    while b:
        a, b = b, a % b
    return a


def pythagorean_triples():
    for z in itertools.count(start=0, step=1):
        for x in range(1, z+1):
            for y in range(x, z+1):
                if x * x + y * y == z * z and reduce(gcd, (x, y, z)) == 1:
                    yield (x, y, z)


def pythagorean_triples_linalg(max_z):
    triples = [np.array([3, 4, 5])]
    lo = 0
    hi = 1

    A = np.array([[-1, 2, 2],
                  [-2, 1, 2],
                  [-2, 2, 3]])

    B = np.array([[1, 2, 2],
                  [2, 1, 2],
                  [2, 2, 3]])

    C = np.array([[1, -2, 2],
                  [2, -1, 2],
                  [2, -2, 3]])

    while hi - lo > 0:
        for triple in triples[lo:hi]:
            for matrix in (A, B, C):
                new_triple = matrix @ triple
                if new_triple[2] < max_z:
                    triples.append(new_triple)
        lo, hi = hi, len(triples)

    return triples


@timed
def test_pythagorean_triples():
    res = []
    for triple in pythagorean_triples():
        if triple[2] > 200:
            break
        res.append(triple)
    return res


@timed
def test_pythagorean_triples_linalg():
    return pythagorean_triples_linalg(max_z=200)


def run_bench(benches, trial_count):
    for bench in benches:
        elapsed = [bench().elapsed for _ in range(trial_count)]
        print(bench.__name__)
        print(" * mean:", np.mean(elapsed))
        print(" * std: ", np.std(elapsed))


run_bench([test_pythagorean_triples, test_pythagorean_triples_linalg], trial_count=10)
