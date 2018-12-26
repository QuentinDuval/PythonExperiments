import functools

@functools.lru_cache(maxsize = None)
def fibo(n):
    if n <= 1:
        return 1
    return fibo(n-1) + fibo(n-2)

# print(fibo(100))

@functools.total_ordering
class Person:
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

    def __eq__(self, other):
        return (self.first_name, self.last_name) == (other.first_name, other.last_name)

    def __lt__(self, other):
        return (self.first_name, self.last_name) < (other.first_name, other.last_name)


"""
Single dispatch is absolutely amazing...
"""

@functools.singledispatch
def serialize(arg, pretty = False):
    return ""

@serialize.register(Person)
def _(person: Person, pretty = False):
    return person.first_name + " " + person.last_name

@serialize.register(list)
def _(l, pretty = False):
    return "[" + ", ".join(serialize(x, pretty) for x in l) + "]"

print(serialize([
    Person(first_name="Quentin", last_name="Duval")
]))
