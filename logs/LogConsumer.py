import abc
from typing import List

from logs.LogEntry import *


class LogConsumer(abc.ABC):
    """
    Abstract base class for any Alerting or Analytics implementation interested in log entries
    """

    # TODO - rename Watcher or something of that kind (or Alert or Rule?)

    @abc.abstractmethod
    def on_log_entry_chunk(self, chunk: List[LogEntry]):
        pass
