import asyncio
import websockets

async def test():
    uri = "ws://localhost:8000/ws/chat"
    async with websockets.connect(uri) as ws:
        await ws.send("this is not valid json")
        response = await ws.recv()
        print(f"Response to malformed input: {response}")

        await ws.send('{"no_message_field": true}')
        response = await ws.recv()
        print(f"Response to missing field: {response}")

asyncio.run(test())