import abc
import sys
import time


"""
Basic interface for any agent acting on an environment, in a given state
"""


class Agent(abc.ABC):
    @abc.abstractmethod
    def get_action(self, env, state):
        pass


"""
Utilities for tracking progress
"""


g_iteration_nb = 0
g_previous_time = None


def print_progress(done, total):
    global g_previous_time, g_iteration_nb
    g_iteration_nb += 1
    current_time = time.time_ns()
    if g_previous_time is not None:
        delay_ms = (current_time - g_previous_time) / 1_000_000
        if delay_ms > 100:
            throughput = g_iteration_nb / delay_ms * 1_000
            sys.stdout.write("\x1b[A") # Clear the line
            sys.stdout.write("{0}/{1} ({2:.2f}%) - {3:.2f} it/s".format(done, total, 100*done/total, throughput))
            sys.stdout.write("\r")
            g_iteration_nb = 0
            g_previous_time = time.time_ns()
    else:
        g_previous_time = time.time_ns()
    

def prange(end_range: int):
    print_progress(0, end_range)
    for i in range(end_range):
        print_progress(i+1, end_range)
        yield i
    sys.stdout.write("\n")
