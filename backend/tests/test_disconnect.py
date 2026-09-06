import asyncio
import json
import websockets

WS_URL = "ws://localhost:8000/ws/chat"


async def test():
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"message": "Tell me about all the membership plans in detail"}))

        # Receive a couple of tokens, then abruptly close the connection mid-stream
        for _ in range(3):
            response = await ws.recv()
            print(f"Received: {response[:80]}")

        print("Closing connection mid-stream now...")
        await ws.close()

    print("Client closed. Check the uvicorn terminal, server should log the disconnect cleanly, no traceback/crash.")


asyncio.run(test())