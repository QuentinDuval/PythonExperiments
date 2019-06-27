"""
Main entry point of the program
"""

from logs.Alerting import *
from logs.Configuration import *
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
    configuration = Configuration(
        poll_interval=10,
        alert_window=120,
        log_file_name="access.log",
        throughput_threshold=10,
        error_threshold=0.5)

    throughput_alerting = ThroughputAlerting(
        throughput_threshold=configuration.throughput_threshold,
        throughput_window_size=configuration.alert_window)

    error_rate_alerting = ErrorRateAlerting(error_rate_threshold=configuration.error_threshold)

    log_parser = W3CLogParser()
    with open(configuration.log_file_name) as log_file:
        file_listener = LogFileStream(log_file)     # TODO - compose with parser ? Or do a wrapper around it...
        time.sleep(configuration.poll_interval)

        while True:
            chunk = []
            for line in file_listener.read_tail():
                print(line)
                log_entry = log_parser.parse(line)
                if log_entry is not None:
                    chunk.append(log_entry)
                    print(log_entry)

            # TODO - the time window is not the same => so you must accumulate entries and get rid of oldest ones
            # TODO - the accumulation must be done by each check ? NO, HAVE an alerting window for all alerts

            print(throughput_alerting.is_high_throughput(chunk))    # TODO - abstract the two alerting behind interface
            print(error_rate_alerting.is_high_error(chunk))         # TODO - abstract the two alerting behind interface

            time.sleep(configuration.poll_interval)  # TODO - not good, this will drift (capture the time at beginning)


if __name__ == '__main__':
    main()

