from dataclasses import dataclass


@dataclass
class Configuration:
    log_file_name: str
    log_poll_interval: int          # Time interval between successive log file polls: TODO - rename
    alert_window: int               # Number of relevant intervals to test alerts on
    throughput_threshold: float
    error_threshold: float
