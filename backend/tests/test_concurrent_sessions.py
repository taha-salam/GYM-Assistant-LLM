import asyncio
import json
import time
import websockets

WS_URL = "ws://localhost:8000/ws/chat"

TEST_MESSAGES = [
    "How much is membership?",
    "What classes are on Saturday?",
    "Who is the yoga trainer?",
    "Can I freeze my membership?",
    "What's the cancellation policy?",
]


async def simulate_user(user_id, message):
    start = time.time()
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"message": message}))

        while True:
            response = await ws.recv()
            data = json.loads(response)
            if data["type"] == "done":
                break
            if data["type"] == "error":
                print(f"[User {user_id}] ERROR: {data['message']}")
                break

    elapsed = time.time() - start
    print(f"[User {user_id}] completed in {elapsed:.2f}s — message: \"{message}\"")


async def main():
    print(f"Launching {len(TEST_MESSAGES)} concurrent simulated users...\n")
    start = time.time()

    tasks = [
        simulate_user(i + 1, msg)
        for i, msg in enumerate(TEST_MESSAGES)
    ]
    await asyncio.gather(*tasks)

    total = time.time() - start
    print(f"\nAll {len(TEST_MESSAGES)} concurrent sessions completed in {total:.2f}s total")


if __name__ == "__main__":
    asyncio.run(main())