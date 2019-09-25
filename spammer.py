"""
Script to spam a server with request, along with a timeout, in order to see how the server reacts to timeout
TODO - try this when the server is behind a load balancer that keeps the connection alive
"""

import asks
import trio
from asks.sessions import Session


class Spammer:
    def __init__(self, nb_concurrent_requests: int, timeout: int):
        self.count = 0
        self.nb_concurrent_requests = nb_concurrent_requests
        self.timeout = timeout
        self.session = Session(connections=nb_concurrent_requests)

    async def spam_server(self, url: str, nb_round: int):
        async with trio.open_nursery() as nursery:
            for _ in range(nb_round * self.nb_concurrent_requests):
                nursery.start_soon(self.get_server, url)

    async def get_server(self, url: str):
        # TODO - check https://trio.readthedocs.io/en/stable/reference-core.html#handling-cancellation
        with trio.move_on_after(self.timeout):
            # time = trio.current_time()
            r = await self.session.get(url)
            self.count += 1
            print(r.text)
            # await trio.sleep_until(time + 1)


spammer = Spammer(nb_concurrent_requests=30, timeout=2)
trio.run(spammer.spam_server, "http://127.0.0.1:8889/threading/wait?millis=1000", 1)
print(spammer.count)
