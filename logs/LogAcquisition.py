"""
Listening to a file being written to
"""

import os


# TODO - acquire the log datetime (the last line) to get the current time?
# TODO - or do we fix the datetime at the beginning ?


class FileStream:
    def __init__(self, file):
        self.file = file
        self.file.seek(0, os.SEEK_END)
        # self.position = self.file.tell()

    def read_tail(self):
        while True:
            # self.position = self.file.tell()
            line = self.file.readline()
            if line:
                yield line
            else:
                # self.file.seek(self.position)
                break
