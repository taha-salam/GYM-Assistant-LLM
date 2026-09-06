import requests
import time
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:1.5b-instruct-q4_K_M"

# A spread of prompt types to see how length/complexity affects performance
TEST_PROMPTS = {
    "short": "What are your gym's opening hours?",
    "medium": "I want to book a fitness class for tomorrow evening. Can you walk me through what classes are available and how I book one?",
    "long": "I'm a new member at your gym and I want to understand everything about your membership plans, including pricing tiers, what each tier includes, whether I can freeze my membership if I travel, how class bookings work, and what happens if I need to cancel. Please explain all of this in detail.",
}

RUNS_PER_PROMPT = 3


def run_single_generation(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
    }

    start_time = time.time()
    first_token_time = None
    token_count = 0
    full_response = ""

    response = requests.post(OLLAMA_URL, json=payload, stream=True)

    for line in response.iter_lines():
        if line:
            chunk = json.loads(line.decode("utf-8"))
            token_text = chunk.get("response", "")

            if token_text and first_token_time is None:
                first_token_time = time.time()

            if token_text:
                token_count += 1
                full_response += token_text

            if chunk.get("done", False):
                break

    end_time = time.time()

    total_time = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time else None
    tokens_per_sec = token_count / total_time if total_time > 0 else 0

    return {
        "total_time_sec": round(total_time, 2),
        "time_to_first_token_sec": round(ttft, 2) if ttft else None,
        "token_count": token_count,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "response_preview": full_response[:100] + "..." if len(full_response) > 100 else full_response,
    }


def main():
    print(f"Benchmarking model: {MODEL}\n")
    all_results = {}

    for label, prompt in TEST_PROMPTS.items():
        print(f"--- Testing '{label}' prompt ---")
        results = []

        for run_num in range(1, RUNS_PER_PROMPT + 1):
            print(f"  Run {run_num}/{RUNS_PER_PROMPT}...")
            result = run_single_generation(prompt)
            results.append(result)
            print(f"    Time to first token: {result['time_to_first_token_sec']}s | "
                  f"Total time: {result['total_time_sec']}s | "
                  f"Tokens/sec: {result['tokens_per_sec']}")

        avg_ttft = round(sum(r["time_to_first_token_sec"] for r in results) / len(results), 2)
        avg_total_time = round(sum(r["total_time_sec"] for r in results) / len(results), 2)
        avg_tokens_per_sec = round(sum(r["tokens_per_sec"] for r in results) / len(results), 2)

        all_results[label] = {
            "avg_time_to_first_token_sec": avg_ttft,
            "avg_total_time_sec": avg_total_time,
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "individual_runs": results,
        }

        print(f"  Averages -> TTFT: {avg_ttft}s | Total: {avg_total_time}s | Tokens/sec: {avg_tokens_per_sec}\n")

    print("=== Summary (for your README) ===")
    for label, data in all_results.items():
        print(f"{label.upper()}: avg TTFT={data['avg_time_to_first_token_sec']}s, "
              f"avg total={data['avg_total_time_sec']}s, "
              f"avg tokens/sec={data['avg_tokens_per_sec']}")

    with open("../docs/benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nFull results saved to docs/benchmark_results.json")


if __name__ == "__main__":
    main()