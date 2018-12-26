"""
Stupid example of yield
"""


def stream_file(file_name):
    with open(file_name) as f:
        for l in f:
            yield l


def gen_filter(p):
    def filter_(gen):
        for x in gen:
            if p(x):
                yield x
    return filter_


def gen_map(f):
    return lambda gen: (f(x) for x in gen)


def pipe(gen, *fs):
    if not fs:
        return gen
    return pipe(fs[0](gen), *fs[1:])


def test_pipeline(file_name):
    for line in pipe(stream_file(file_name), gen_map(lambda x: x.strip()), gen_filter(lambda x: "yield" in x)):
        print(line)


test_pipeline("/Users/duvalquentin/PycharmProjects/Experiments/Yiedling.py")

