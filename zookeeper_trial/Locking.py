from kazoo.client import KazooClient
import os


class TestClient:
    # TODO - lack the support for Zookeeper falling

    def __init__(self):
        self.acquired = {}

    def loop(self):
        while True:
            is_exit = self.ask_input()
            if is_exit:
                break
        self.release_all()

    def ask_input(self):
        command = input("object>")
        if command == "exit":
            return True

        if command == "ls":
            self.list_acquired_locks()
            return False

        if any(not c.isdigit() for c in command):
            print("invalid object id")
            return False

        object_id = int(command)
        if object_id in self.acquired:
            self.release(object_id, self.acquired[object_id])
            del self.acquired[object_id]
        else:
            self.acquire(object_id)

    def list_acquired_locks(self):
        pass

    def acquire(self, object_id):
        lock = zk.Lock("/object/{object_id}".format(object_id=object_id), str(os.getpid()))
        if lock.acquire(blocking=False, ephemeral=True):
            self.acquired[object_id] = lock
            print("Object {object_id} locked".format(object_id=object_id))
        else:
            print("Object {object_id} is already locked".format(object_id=object_id))

    def release(self, object_id, lock):
        lock.release()
        print("Object {object_id} release".format(object_id=object_id))

    def release_all(self):
        for object_id, lock in self.acquired.items():
            self.release(object_id, lock)


if __name__ == '__main__':
    zk = KazooClient(hosts='127.0.0.1:2181')
    try:
        zk.start()
        client = TestClient()
        client.loop()
    finally:
        zk.stop()
