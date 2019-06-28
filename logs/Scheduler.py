import sched
import time


def schedule_every(time_interval: float, to_repeat: 'function to call repeatedly'):
    """
    Wraps a function to call it repeatedly, every "time_interval" seconds
    """
    # TODO - use module sched ?
    # https://stackoverflow.com/questions/474528/what-is-the-best-way-to-repeatedly-execute-a-function-every-x-seconds-in-python

    time.sleep(time_interval)
    while True:
        start_time = time.time()
        end_time = time.time()
        to_repeat()
        processing_time = end_time - start_time
        print(processing_time)
        time.sleep(time_interval - processing_time)

    # TODO - find a way to avoid the drift better? Like we could take into account the date in the logs
