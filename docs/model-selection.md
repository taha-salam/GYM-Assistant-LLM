# Model Selection

## Models Considered

We tested two CPU-friendly, instruction-tuned models from the Qwen2.5 family, both at Q4 quantization, run locally through Ollama:

1. qwen2.5:3b-instruct-q4_K_M (3 billion parameters)
2. qwen2.5:1.5b-instruct-q4_K_M (1.5 billion parameters)

Both fall within the assignment's required range of 0.5B to 4B parameters at Q4 quantization.

## Why We Tested Both

The 3B model is generally expected to give better response quality since it has more parameters, but it also needs more compute per token, which can slow it down badly on CPU-only hardware. Since this assignment requires fully local, CPU-based inference with real-time streaming, speed is not optional, it directly affects whether the chatbot feels usable. We decided to benchmark both models on our own hardware before picking one, rather than assuming the bigger model was automatically the better choice.

## How We Benchmarked

We wrote a script that sends three types of prompts to each model: a short question, a medium-length question, and a long, detailed question. Each prompt type was run three times, and we averaged the results to reduce the effect of any single slow or fast run. For each run we measured:

- Time to first token (how long the user waits before seeing anything appear on screen)
- Total response time (how long the full answer takes to finish generating)
- Tokens per second (how fast the model generates text once it starts)

## Results

### qwen2.5:3b-instruct-q4_K_M

| Prompt Type | Avg Time to First Token | Avg Total Time | Avg Tokens/sec |
|---|---|---|---|
| Short | 4.71s | 13.47s | 5.74 |
| Medium | 3.19s | 89.48s | 7.37 |
| Long | 3.42s | 114.62s | 8.45 |

### qwen2.5:1.5b-instruct-q4_K_M

| Prompt Type | Avg Time to First Token | Avg Total Time | Avg Tokens/sec |
|---|---|---|---|
| Short | 3.52s | 11.52s | 11.04 |
| Medium | 2.54s | 28.15s | 15.85 |
| Long | 2.74s | 25.89s | 11.80 |

## Decision

We chose qwen2.5:1.5b-instruct-q4_K_M for the full project.

The 1.5B model was roughly twice as fast as the 3B model across every prompt type, both in tokens per second and in total response time. For medium and long prompts, the difference was especially large, the 3B model took over a minute and a half on average for a medium-length reply, while the 1.5B model finished in under 30 seconds.

We decided this tradeoff was worth it because a gym membership assistant does not need heavy reasoning. Its job is mostly straightforward: answering questions about class schedules, membership plans, trainers, and handling booking or freezing requests. This is closer to structured information lookup and policy-following than complex, multi-step reasoning, so a smaller model can handle it well if the system prompt and conversation design are done carefully. Given that real-time behavior and performance make up a meaningful part of the grading criteria, the speed advantage of the 1.5B model matters more here than the small quality gap it may have compared to the 3B model.

## Known Limitation

Even with the 1.5B model, total response time for medium and long replies is still around 25 to 30 seconds on our hardware, which is not instantaneous. Streaming the response token by token helps a lot here, since the user starts seeing text after roughly 2.5 to 3.5 seconds rather than waiting for the full reply, but it is worth being upfront that this is a CPU-constrained system, not a low-latency cloud API. We consider this an acceptable tradeoff given the assignment's requirement to run fully locally on CPU.