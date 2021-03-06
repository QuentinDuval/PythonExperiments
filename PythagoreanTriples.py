from collections import *
from functools import *
import itertools
import numpy as np
import timeit


# TODO - compare with https://aras-p.info/blog/2018/12/28/Modern-C-Lamentations/


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
    for z in itertools.count():
        for x in range(1, z+1):
            for y in range(x, z+1):
                if x * x + y * y == z * z and reduce(gcd, (x, y, z)) == 1:
                    yield (x, y, z)


def pythagorean_triples_linalg(max_z):

    # TODO - transform this to use YIELD (co-routine which accepts values as well)

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


def pythagorean_triples_linalg_2(max_z):

    # TODO - transform this to use YIELD (co-routine which accepts values as well)

    matrices = np.array(
        [[-1, 2, 2],
         [-2, 1, 2],
         [-2, 2, 3],
         [1, 2, 2],
         [2, 1, 2],
         [2, 2, 3],
         [1, -2, 2],
         [2, -1, 2],
         [2, -2, 3]
    ])

    # Or you could use:
    # matrices = np.concatenate((A, B, C), axis=0)

    triples = [np.array([3, 4, 5])]
    lo, hi = 0, 1
    while hi - lo > 0:
        for triple in triples[lo:hi]:
            new_triples = matrices @ triple # TODO - zarb... il faudrait transposer normalement
            new_triples = np.reshape(new_triples, (3,3))
            for new_triple in new_triples:
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


@timed
def test_pythagorean_triples_linalg_2():
    return pythagorean_triples_linalg_2(max_z=200)


def next_pythagorean_triples(previous):
    matrices = np.array(
        [[-1, 2, 2],
         [-2, 1, 2],
         [-2, 2, 3],
         [1, 2, 2],
         [2, 1, 2],
         [2, 2, 3],
         [1, -2, 2],
         [2, -1, 2],
         [2, -2, 3]])

    next_triples = np.transpose(matrices @ np.transpose(previous))
    next_triples = next_triples.reshape((3 * previous.shape[0], previous.shape[1]))
    return next_triples


def pythagorean_triples_by_stage():
    # TODO - filtering based on max_z?
    current = np.array([[3, 4, 5]])
    yield current
    while True:
        current = next_pythagorean_triples(current)
        yield current


@timed
def test_pythagorean_triples_linalg_3():
    for depth, triples in enumerate(pythagorean_triples_by_stage()):
        if depth >= 5:
            break
        print(triples)


def run_bench(benches, trial_count):
    for bench in benches:
        elapsed = [bench().elapsed for _ in range(trial_count)]
        print(bench.__name__)
        print(" * res: ", len(bench().result))
        print(" * mean:", np.mean(elapsed))
        print(" * std: ", np.std(elapsed))


# run_bench([test_pythagorean_triples, test_pythagorean_triples_linalg, test_pythagorean_triples_linalg_2], trial_count=10)

"""
for triangle in test_pythagorean_triples_linalg_2().result:
    print(triangle)
"""

test_pythagorean_triples_linalg_3()
