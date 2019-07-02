from kazoo.client import KazooClient
import os


class TestClient:
    # TODO - lack the support for Zookeeper falling

    def __init__(self):
        self.acquired = {}
        self.keep_looping = True

    def loop(self):
        while self.keep_looping:
            self.ask_input()
        self.release_all()

    def ask_input(self):
        command = input("object>")
        if command == "exit":
            self.keep_looping = False
            return

        if command == "reserve":
            self.reserve_id()
            return

        if command == "ls":
            self.list_acquired_locks()
            return

        if any(not c.isdigit() for c in command):
            print("invalid object id")
            return

        object_id = int(command)
        if object_id in self.acquired:
            self.release(object_id, self.acquired[object_id])
            del self.acquired[object_id]
        else:
            self.acquire(object_id)

    def reserve_id(self):
        counter = zk.Counter("/id", default=1)
        counter += 1
        print(counter.value)

    def list_acquired_locks(self):
        children = zk.get_children(path="/object")
        print(children) # TODO - problem here: we get children that are unlocked

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

    def on_zookeeper_status_update(self, state):
        # print(state)
        # self.keep_looping = False
        pass


if __name__ == '__main__':
    client = TestClient()
    zk = KazooClient(hosts='127.0.0.1:2181')
    try:
        zk.add_listener(lambda zk_state: client.on_zookeeper_status_update(zk_state))
        zk.start()
        client.loop()
    finally:
        zk.stop()
