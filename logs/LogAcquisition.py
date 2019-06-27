"""
Listening to a file being written to
"""

import os


# TODO - acquire the log datetime (the last line) to get the current time?
# TODO - or do we fix the datetime at the beginning ?


class LogFileStream:
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

    '''
    # TODO - rather than this, just compose with the with(open)?

    def __enter__(self):
        self.file = open(self.file_path, 'r')
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.file.close()
        return False    # Do not ignore exceptions
    '''

