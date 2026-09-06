import asyncio
import json
import sys
import websockets

async def test(session_id_to_use=None):
    uri = "ws://localhost:8000/ws/chat"
    async with websockets.connect(uri) as ws:
        message_payload = {"message": "What did I just ask you?"}
        if session_id_to_use:
            message_payload["session_id"] = session_id_to_use

        await ws.send(json.dumps(message_payload))

        session_id = None
        while True:
            response = await ws.recv()
            data = json.loads(response)

            if data["type"] == "session_start":
                session_id = data["session_id"]
                print(f"[Session started: {session_id}]")
            elif data["type"] == "token":
                print(data["content"], end="", flush=True)
            elif data["type"] == "done":
                print()
                break
            elif data["type"] == "error":
                print(f"[ERROR: {data['message']}]")
                break

    return session_id

if __name__ == "__main__":
    session_id_arg = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(test(session_id_arg))