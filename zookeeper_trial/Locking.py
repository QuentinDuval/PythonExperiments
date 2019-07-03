from kazoo.client import KazooClient
import os
import time


class TestClient:
    # TODO - lack the support for Zookeeper falling

    def __init__(self, reservation_size):
        self.acquired = {}
        self.keep_looping = True
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
            self.reserve_id()
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

        children = zk.get_children("/leader")
        if not children:
            election = zk.Election("/leader", str(os.getpid()))
            election.run(when_elected)
        else:
            print("Leader already exists", children[0])

    def reserve_id(self):
        counter = zk.Counter("/id", default=1)
        counter += 1
        print(counter.value)

    def list_acquired_locks(self):
        # ZK will keep listing as children released locks (need this filter)
        locked_children = [child for child in zk.get_children(path="/object") if self.is_locked(child)]
        print(locked_children)

    def is_locked(self, lock_id):
        _, stats = zk.get(path="/object/{child}".format(child=lock_id))
        return stats.numChildren != 0

    def lock_object(self, lock_id):
        if any(not c.isdigit() for c in lock_id):
            print("invalid object id")
            return

        object_id = int(lock_id)
        if object_id in self.acquired:
            self.release(object_id, self.acquired[object_id])
            del self.acquired[object_id]
        else:
            self.acquire(object_id)

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
    zk = KazooClient(hosts='127.0.0.1:2181')
    client = TestClient(reservation_size=1)
    try:
        zk.add_listener(lambda zk_state: client.on_zookeeper_status_update(zk_state))
        zk.start()
        client.loop()
    finally:
        zk.stop()
