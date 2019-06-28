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


# TODO - use argparse to avoid this pesky configuration from JSON... or add the name of the configuration in argument


def read_configuration():
    configuration = Configuration(
        log_file_name="tmp/access.log",
        log_poll_interval=10,
        alert_window=12,
        throughput_threshold=10,
        error_threshold=0.5)

    try:
        with open('configuration.json') as config_file:
            data = json.load(config_file)
            configuration.log_file_name = data["log_file_name"]
            configuration.throughput_threshold = data["throughput_threshold"]
    except FileNotFoundError:
        pass    # Just use default configuration in that case
    return configuration


class Monitor:
    def __init__(self, config: Configuration):
        self.config = config
        self.throughput_alerting = ThroughputAlerting(threshold=config.throughput_threshold, window_size=config.alert_window)
        self.error_rate_alerting = ErrorRateAlerting(error_rate_threshold=config.error_threshold)
        self.log_parser = ApacheCommonLogParser()

    def start(self):
        with open(self.config.log_file_name) as log_file:
            file_listener = FileStream(log_file)
            schedule_every(self.config.log_poll_interval, lambda: self.poll(file_listener))

    def poll(self, file_listener):
        chunk = []
        for line in file_listener.read_tail():
            log_entry = self.log_parser.parse(line)
            if log_entry is not None:
                chunk.append(log_entry)

        print(self.throughput_alerting.is_high_throughput(chunk))       # TODO - abstract the two alerting behind interface
        print(self.error_rate_alerting.is_high_error(chunk))            # TODO - abstract the two alerting behind interface
        print(most_impacted_sessions(chunk))                            # TODO - we can also abstract the statistics


def main():
    scheduler = Monitor(config=read_configuration())
    scheduler.start()


if __name__ == '__main__':
    main()

