import math


def drop_egg(n: int) -> int:
    return math.ceil(math.sqrt(2 * n + 0.25) - 0.5)


print(drop_egg(100))
print(drop_egg(200))
print(drop_egg(300))
print(drop_egg(400))
print(drop_egg(500))
