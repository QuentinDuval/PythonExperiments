from collections import OrderedDict
import itertools
import time


"""
Registering some functions using decorators
"""

promotions = []

def promotion(promo):
    promotions.append(promo)
    return promo

@promotion
def basic_promo(prices):
    return sum(p * 0.9 for p in prices)

@promotion
def lowest_price_refund(prices):
    min_price = min(prices)
    return sum(prices) - min_price

def best_promo(prices):
    return min(promo(prices) for promo in promotions)

# print(best_promo([5.5, 7.5, 6]))


"""
Printing argument and return of a function (good for debugging)
"""

def traced(f):
    def traced_f(*args, **kwargs):
        start_time = time.clock()
        result = f(*args, **kwargs)
        end_time = time.clock()
        running_time = (end_time - start_time) * 1000
        arg_to_str = (repr(a) for a in args)
        karg_to_str = (str(k) + "=" + repr(v) for k, v in kwargs.items())
        print("[TRACE]", f.__name__ + "(" + ",".join(itertools.chain(arg_to_str, karg_to_str)) + ") =>", result, "in", running_time, "ms")
        return result
    return traced_f

@traced
def hello(name):
    return "Hello, " + name + "!"

# print(hello("my friend"))


"""
Adding a cache around a function
=> Using partial application (currying manually) to pass parameters
"""

def with_cache(max_size):
    def decorator(f):
        cache = OrderedDict()
        def cached_f(*args): # TODO - make it work with kwarg
            key = args
            val = cache.get(key, None)
            if val is not None:
                return val

            val = f(*args)
            cache[key] = val
            if max_size and len(cache) > max_size:
                cache.popitem()

            return val
        return cached_f
    return decorator

@with_cache(max_size=None)
def fib(n):
    if n <= 1: return 1
    return fib(n-1) + fib(n-2)

# print(fib(100))


"""
Registering into an object (not a singleton... just a global variable)
"""

class PromoRegistry:
    def __init__(self):
        self.promotions = []

    def register(self, f):
        self.promotions.append(f)
        return f

    def __len__(self):
        return self.promotions

    def __getitem__(self, item):
        return self.promotions[item]

    def __str__(self):
        return str(self.promotions)

christmas_promotions = PromoRegistry()

@christmas_promotions.register
def family_promo(prices):
    return sum(prices)

# print(christmas_promotions)


"""
And you can obviously create an object using a decorator, whose name will allow you to register more things!
(This is the basic design behind @functools.singledispatch)
"""

def multi_dispatch(default_impl):
    class MultiDispatcher:
        def __init__(self):
            self.options = {}
            self.default_impl = default_impl

        def register(self, *argument_types):
            def decorator(f):
                self.options[argument_types] = f
                return f
            return decorator

        def __call__(self, *args):
            arg_types = tuple(type(a) for a in args)
            f = self.options.get(arg_types, self.default_impl)
            return f(*args)
    return MultiDispatcher()

@multi_dispatch
def collide(a, b):
    print("Boom!", a, "and", b)

@collide.register(int, int)
def _(a, b):
    print("Collision:", a + b)

@collide.register(int, str)
def _(a, b):
    print("Collision:", a * b)

# collide(1, 2)
# collide(5, "b")
# collide("a", "b")
