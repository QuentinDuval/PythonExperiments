"""
Script to spam a server with request, along with a timeout, in order to see how the server reacts to timeout
"""

import asks
import trio
from asks.sessions import Session


class Spammer:
    def __init__(self, timeout: int, keep_connections: bool = False):
        self.count = 0
        self.timeout = timeout
        self.keep_connections = keep_connections
        self.session = None

    async def spam_server(self, url: str, rounds: int):
        self.session = Session(connections=max(rounds)) if self.keep_connections else None
        async with trio.open_nursery() as nursery:
            for req_by_round in rounds:
                time = trio.current_time()
                for _ in range(req_by_round):
                    nursery.start_soon(self.get_server, url)
                await trio.sleep_until(time + 1)
        self.session = None

    async def get_server(self, url: str):
        with trio.move_on_after(self.timeout):
            # print("Request...", trio.current_time())
            if not self.session:
                r = await asks.get(url)
            else:
                r = await self.session.get(url)
            self.count += 1
            print(r.text)


spammer = Spammer(timeout=2, keep_connections=True)
trio.run(spammer.spam_server, "http://127.0.0.1:8889/threading/wait?millis=1000", [30, 10, 10])
print(spammer.count)


# TODO - the usual core focus is normally to make sure the time by request stays constant under load
# TODO - try this when the server is behind a load balancer that keeps the connection alive
# TODO - try to spam the server and check its memory consumption (and the GC might make it slower)
# TODO - compute the probability of failure with multiple services
