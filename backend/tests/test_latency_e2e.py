import asyncio
import json
import time
import websockets

WS_URL = "ws://localhost:8000/ws/chat"

# One message per routing path, so we can see the real latency profile of
# each path separately, not just an average that hides the difference.
TEST_CASES = {
    "gym_keyword_shortcut (skips classifier)": "How much is the membership?",
    "classifier_routed_ambiguous": "Hmm, what should I do this weekend?",
    "medical_redirect": "My dog bit me, what do I do?",
    "off_topic_redirect": "What's the weather like today?",
    "recall_deterministic": "What was my last message?",
    "closing_deterministic": "ok bye",
}

RUNS_PER_CASE = 3


async def run_single_message(message, session_id=None):
    start = time.time()
    first_token_time = None
    token_count = 0

    async with websockets.connect(WS_URL) as ws:
        payload = {"message": message}
        if session_id:
            payload["session_id"] = session_id

        await ws.send(json.dumps(payload))

        current_session_id = session_id
        while True:
            response = await ws.recv()
            data = json.loads(response)

            if data["type"] == "session_start":
                current_session_id = data["session_id"]
            elif data["type"] == "token":
                if first_token_time is None:
                    first_token_time = time.time()
                token_count += 1
            elif data["type"] == "done":
                break
            elif data["type"] == "error":
                print(f"  ERROR: {data['message']}")
                break

    end = time.time()
    total_time = end - start
    ttft = (first_token_time - start) if first_token_time else None

    return total_time, ttft, current_session_id


async def main():
    print("End-to-end latency test (through full WebSocket pipeline, not raw Ollama)\n")
    print(f"Each case run {RUNS_PER_CASE} times, using a fresh session each time ")
    print("so history/context doesn't affect timing.\n")

    for label, message in TEST_CASES.items():
        print(f"--- {label} ---")
        print(f'    Message: "{message}"')

        total_times = []
        ttfts = []

        for run_num in range(1, RUNS_PER_CASE + 1):
            total_time, ttft, _ = await run_single_message(message)
            total_times.append(total_time)
            if ttft is not None:
                ttfts.append(ttft)
            ttft_display = f"{ttft:.2f}s" if ttft is not None else "N/A (no streamed tokens)"
            print(f"    Run {run_num}: total={total_time:.2f}s, TTFT={ttft_display}")

        avg_total = sum(total_times) / len(total_times)
        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else None
        avg_ttft_display = f"{avg_ttft:.2f}s" if avg_ttft is not None else "N/A"
        print(f"    Average: total={avg_total:.2f}s, TTFT={avg_ttft_display}\n")


if __name__ == "__main__":
    asyncio.run(main())