import bisect
import uuid
from collections import defaultdict

import pandas as pd


class ConsistentHashingLoadBalancer:
    def __init__(self, ring_size: int, hash_count: int):
        self.ring_size = ring_size
        self.servers = []
        self.vnode_to_server = defaultdict(list)
        self.salt = [str(i) for i in range(hash_count)]

    def add_server(self, server_id: str):
        """
        To handle conflict we use a list: first one coming, first served.
        But we keep the other node: if the first one disappears, it gets the load
        """
        for salt in self.salt:
            vnode_id = self._hash(server_id + salt)
            self.vnode_to_server[vnode_id].append(server_id)
            bisect.insort_left(self.servers, vnode_id)

    def remove_server(self, server_id: str):
        for salt in self.salt:
            vnode_id = self._hash(server_id + salt)
            self.vnode_to_server[vnode_id].remove(server_id)
            pos = bisect.bisect_left(self.servers, vnode_id)
            self.servers.pop(pos)

    def serve(self, request_id: str) -> str:
        pos = bisect.bisect_left(self.servers, self._hash(request_id))
        vnode_id = self.servers[pos] if pos < len(self.servers) else self.servers[0]
        return self.vnode_to_server[vnode_id][0]

    def _hash(self, input) -> int:
        return hash(input) % self.ring_size

    def servers_positions(self):
        output = []
        for position in self.servers:
            output.append((position, self.vnode_to_server[position]))
        return output

    def servers_share(self):
        shares = {}
        prev_pos = self.servers[-1] - self.ring_size
        for pos in self.servers:
            server = self.vnode_to_server[pos][0]
            shares[server] = shares.get(server, 0) + (pos - prev_pos)
            prev_pos = pos
        return shares


consistent_hashing = ConsistentHashingLoadBalancer(ring_size=4096, hash_count=10)
for i in range(20):
    consistent_hashing.add_server(str(i))

shares = consistent_hashing.servers_share()
df = pd.DataFrame(data=list(shares.values()), columns=["slots"])
df['percentage'] = (df['slots'] / consistent_hashing.ring_size) * 100
print(df.describe())

counts = defaultdict(int)
for _ in range(100000):
    request_id = uuid.uuid4().hex
    counts[consistent_hashing.serve(request_id)] += 1

df = pd.DataFrame(data=list(counts.values()), columns=["hits"])
print(df.describe())
