from kazoo.client import KazooClient
import os
import time


class IdentifierStream:
    def __init__(self, zk, reservation_size):
        self.zk = zk
        self.reservation_size = reservation_size
        self.current_id = None
        self.last_id = None
        self.counter = zk.Counter("/id", default=1)

    def __next__(self):
        # TODO - race condition here if we use counter.value? how does it work in behind?
        if self.current_id is None or self.current_id >= self.last_id:
            self.counter += self.reservation_size
            self.current_id = self.counter.pre_value + 1
            self.last_id = self.counter.post_value
        else:
            self.current_id += 1
        return self.current_id


class TestClient:
    def __init__(self, zk, reservation_size):
        self.zk = zk
        self.acquired = {}
        self.keep_looping = True
        self.id_stream = IdentifierStream(zk, reservation_size)
        self.reservation_size = reservation_size

    def loop(self):
        while self.keep_looping:
            self.ask_input()
        self.release_all()

    def ask_input(self):
        command = input("command>")
        if command == "exit":
            self.keep_looping = False
        elif command == "reserve":
            print(self.reserve_id())
        elif command == "leader":
            self.elect_leader()
        elif command == "ls":
            self.list_acquired_locks()
        elif command.startswith("lock"):
            lock_id = command[len("lock")+1:]
            self.lock_object(lock_id)
        else:
            print("Unknown command")

    def elect_leader(self):
        def when_elected():
            print("Taking the lead for 10s...")
            time.sleep(10)
            print("Dropping the lead now...")

        children = self.zk.get_children("/leader")
        if not children:
            election = self.zk.Election("/leader", str(os.getpid()))
            election.run(when_elected)
        else:
            print("Leader already exists", children[0])

    def reserve_id(self):
        return next(self.id_stream)

    def list_acquired_locks(self):
        # ZK will keep listing as children released locks (need this filter)
        locked_children = [child for child in self.zk.get_children(path="/object") if self.is_locked(child)]
        print(locked_children)

    def is_locked(self, lock_id):
        _, stats = self.zk.get(path="/object/{child}".format(child=lock_id))
        return stats.numChildren != 0

    def lock_object(self, lock_id):
        if any(not c.isdigit() for c in lock_id):
            print("invalid object id")
            return

        object_id = int(lock_id)
        if object_id in self.acquired:
            self.release(object_id)
            print("Object {object_id} release".format(object_id=object_id))
        else:
            if self.acquire(object_id):
                print("Object {object_id} locked".format(object_id=object_id))
            else:
                print("Object {object_id} is already locked".format(object_id=object_id))

    def acquire(self, object_id):
        lock = self.zk.Lock("/object/{object_id}".format(object_id=object_id), str(os.getpid()))
        if lock.acquire(blocking=False, ephemeral=True):
            self.acquired[object_id] = lock
            return True
        return False

    def release(self, object_id):
        self.acquired[object_id].release()
        del self.acquired[object_id]

    def release_all(self):
        for object_id, lock in self.acquired.items():
            lock.release()
        self.acquired.clear()

    def on_zookeeper_status_update(self, state):
        # TODO - lack the support for Zookeeper falling ?
        # print(state)
        # self.keep_looping = False
        pass


def main():
    zk = KazooClient(hosts='127.0.0.1:2181')
    client = TestClient(zk, reservation_size=5)
    try:
        zk.add_listener(lambda zk_state: client.on_zookeeper_status_update(zk_state))
        zk.start()
        client.loop()
    finally:
        zk.stop()


if __name__ == '__main__':
    main()
