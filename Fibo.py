def fibo(n):
    curr, next = 0, 1
    for i in range(0, n):
        curr, next = next, curr + next
    return curr


import time
start_time = time.time()
for i in range(0, 1000):
    fibo(1000)
print("--- %s micro-seconds ---" % (1000 * (time.time() - start_time)))