from concurrent.futures import ThreadPoolExecutor, wait
from kazoo.client import KazooClient
from zookeeper_trial.Locking import TestClient
import numpy as np
import time


def go_baby_go_locky(client: TestClient, task_size: int):
    locked = set()
    for object_id in np.random.randint(1, 1000, size=task_size):
        if object_id in locked:
            client.release(object_id)
            locked.remove(object_id)
        else:
            if client.acquire(object_id):
                locked.add(object_id)


def go_baby_go_incry(client: TestClient, task_size: int):
    for _ in range(task_size):
        client.reserve_id()


def main():
    zk = KazooClient(hosts='127.0.0.1:2181')
    client = TestClient(zk, reservation_size=1)
    try:
        zk.add_listener(lambda zk_state: client.on_zookeeper_status_update(zk_state))
        zk.start()

        workers = 100
        tasks = 1000
        task_size = 10
        start = time.time()

        print("Start {workers} workers with {tasks} tasks".format(workers=workers, tasks=tasks))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for _ in range(tasks):
                future = pool.submit(lambda: go_baby_go_locky(client, task_size))   # reach easily 1000 req / s (increases with number of workers)
                # future = pool.submit(lambda: go_baby_go_incry(client, task_size)) # decreases with number of workers (due to retries)
                futures.append(future)
            print(futures)
            wait(futures)

        end = time.time()
        duration = end - start
        total_request = tasks * task_size
        print("Took {duration} seconds for {request} requests ({throughput} request/s)".format(
            duration=duration, request=total_request, throughput=total_request/duration
        ))

    finally:
        zk.stop()


if __name__ == '__main__':
    main()
