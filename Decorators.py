from collections import OrderedDict
import itertools


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

print(best_promo([5.5, 7.5, 6]))


"""
Printing argument and return of a function (good for debugging)
"""

def traced(f):
    def traced_f(*args, **kwargs):
        result = f(*args, **kwargs)
        arg_to_str = (repr(a) for a in args)
        karg_to_str = (str(k) + "=" + repr(v) for k, v in kwargs.items())
        print(f.__name__ + "(" + ",".join(itertools.chain(arg_to_str, karg_to_str)) + ") =>", result)
        return result
    return traced_f

@traced
def hello(name):
    return "Hello, " + name + "!"

print(hello("my friend"))


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

print(fib(100))
