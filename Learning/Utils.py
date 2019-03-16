import random


def join_split(xs, ys, ratio, seed=None):
    def unzip(zipped):
        return [list(x) for x in zip(*zipped)]

    zipped = list(zip(xs, ys))

    rng = random.Random(seed=seed) if seed else random.Random()
    rng.shuffle(zipped)

    k = int(len(xs) * ratio)
    return unzip(zipped[:k]), unzip(zipped[k:])
