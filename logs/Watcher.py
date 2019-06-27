"""
Main entry point of the program
"""

from logs.Alerting import *
from logs.LogAcquisition import *
from logs.LogParser import *

import time


# - Read the configuration
# - Use asyncIO to read the file from time to time
#   (every 10s... but how do you check it does not drift? could have fixed windows...)
#   (other possibility is to fixed 10s like 0, 10, 20, 30... and just do a kind of "modulo" to group)
# - Each time your read it, compute the statistics and status alert
# - Print these information on the screen


# TODO


def main():
    alerting = ThroughputAlerting(throughput_threshold=10, throughput_window_size=10)
    log_parser = W3CLogParser()
    with open("access.log") as log_file:
        file_listener = LogFileStream(log_file)     # TODO - compose with parser ? Or do a wrapper around it...

        while True:
            chunk = []
            for line in file_listener.read_tail():
                print(line)
                log_entry = log_parser.parse(line)
                if log_entry is not None:
                    chunk.append(log_entry)

            # TODO - the time window is not the same => so you must accumulate entries and get rid of oldest ones
            print(alerting.get_status(chunk))

            time.sleep(10)  # TODO - not good, this will drift


if __name__ == '__main__':
    main()

