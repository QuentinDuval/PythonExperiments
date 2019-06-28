"""
Main entry point of the program
"""

from logs.Analytics import *
from logs.Alerting import *
from logs.Configuration import *
from logs.LogAcquisition import *
from logs.LogParser import *
from logs.Scheduler import *

import json


# - Read the configuration
# - Use asyncIO to read the file from time to time
#   (every 10s... but how do you check it does not drift? could have fixed windows...)
#   (other possibility is to fixed 10s like 0, 10, 20, 30... and just do a kind of "modulo" to group)
# - Each time your read it, compute the statistics and status alert
# - Print these information on the screen


def read_configuration():
    configuration = Configuration(
        log_file_name="tmp/access.log",
        log_poll_interval=10,
        alert_window=12,
        throughput_threshold=10,
        error_threshold=0.5)

    with open('configuration.json') as config_file:
        data = json.load(config_file)
        configuration.log_file_name = data["log_file_name"]
        configuration.throughput_threshold = data["throughput_threshold"]

    return configuration


class Monitor:
    def __init__(self, config: Configuration):
        self.config = config
        self.throughput_alerting = ThroughputAlerting(threshold=config.throughput_threshold, window_size=config.alert_window)
        self.error_rate_alerting = ErrorRateAlerting(error_rate_threshold=config.error_threshold)
        self.log_parser = ApacheCommonLogParser()

    def start(self):
        # TODO - better error in case the file does not exist

        with open(self.config.log_file_name) as log_file:
            file_listener = FileStream(log_file)
            schedule_every(self.config.log_poll_interval, lambda: self.poll(file_listener))

    def poll(self, file_listener):
        chunk = []
        for line in file_listener.read_tail():
            log_entry = self.log_parser.parse(line)
            if log_entry is not None:
                chunk.append(log_entry)

        print(self.throughput_alerting.is_high_throughput(chunk))    # TODO - abstract the two alerting behind interface
        print(self.error_rate_alerting.is_high_error(chunk))         # TODO - abstract the two alerting behind interface
        print(most_impacted_sessions(chunk))


def main():
    scheduler = Monitor(config=read_configuration())
    scheduler.start()


if __name__ == '__main__':
    main()

