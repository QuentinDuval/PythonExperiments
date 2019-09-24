import asks
import trio

asks.init(trio)

# Taken from
#   https://medium.com/@pgjones/hypercorn-is-now-a-trio-asgi-server-2e198898c08f


class App:
    def __init__(self, scope):
        self.scope = scope

    async def __call__(self, receive, send):
        while True:
            event = await receive()
            if (
                    event['type'] == 'http.request' and
                    not event.get('more_body', False)
            ):
                if self.scope['path'] == '/':
                    await self.sleep_and_send(send)
                elif self.scope['path'] == '/time':
                    await self.send_time(send)
                else:
                    await self.send_404(send)
            elif event['type'] == 'http.disconnect':
                break
            elif event['type'] == 'lifespan.startup':
                await send({'type': 'lifespan.startup.complete'})
            elif event['type'] == 'lifespan.cleanup':
                await send({'type': 'lifespan.cleanup.complete'})
            elif event['type'] == 'lifespan.shutdown':
                await send({'type': 'lifespan.shutdown.complete'})

    async def sleep_and_send(self, send):
        await trio.sleep(1)
        await send({
            'type': 'http.response.start',
            'status': 200,
            'headers': []
        })
        await send({
            'type': 'http.response.body',
            'more_body': False
        })

    async def send_time(self, send):
        response = await asks.get('http://worldclockapi.com/api/json/utc/now')
        data = response.json()['currentDateTime'].encode('ascii')
        await send({
            'type': 'http.response.start',
            'status': 200,
            'headers': [(b'content-length', b"%d" % len(data))],
        })
        await send({
            'type': 'http.response.body',
            'body': data,
            'more_body': False,
        })

    async def send_404(self, send):
        await send({
            'type': 'http.response.start',
            'status': 404,
            'headers': [(b'content-length', b'0')],
        })
        await send({
            'type': 'http.response.body',
            'body': b'',
            'more_body': False,
        })
