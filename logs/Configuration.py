# Read the configuration:
# - poll interval (default 10s)
# - alert window (default 2 min)
# - log file name (default "/tmp/access.log")
# - high traffic threshold (default 10/s)
# - log format parser (default is W3C)

from dataclasses import dataclass


@dataclass
class Configuration:
    poll_interval: int
    alert_window: int
    log_file_name: str
    throughput_threshold: float
    error_threshold: float
