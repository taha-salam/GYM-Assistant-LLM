# AI Prompt Log

This log tracks every significant prompt used with AI tools (Claude) during the development of FitBot, per the assignment's requirement to explain and reproduce prompts during the viva. Entries are in chronological order, grouped by development phase.

---

## Setup & Planning

**Prompt:** "Just explain me how to do this assignment. do not do it right now"
**Purpose:** Get a full phase-by-phase breakdown of the assignment before writing any code, to plan the approach.

**Prompt:** "Gym / Fitness Membership Assistant... My selected case this. Now divide evenly this assignment into 2 persons so we can start" (later corrected to working solo)
**Purpose:** Confirm domain selection and get a task breakdown, later adjusted to a solo sequential plan.

**Prompt:** "lets just start i am working alone. where do i start from and how do i start. Give me steps one by one"
**Purpose:** Get a concrete, sequential setup plan (repo structure, venv, Ollama, benchmarking, conversation manager, backend, frontend, testing).

**Prompt:** "lets start with step-1 in detail step by step"
**Purpose:** Get exact PowerShell commands for repo/folder/venv setup on Windows.

---

## Environment Troubleshooting

**Prompt:** Pasted a Python-not-found error after running `python -m venv venv`
**Purpose:** Diagnose and fix a Windows App Execution Alias issue blocking the `python` command.

**Prompt:** Pasted continued errors after using `py` instead of `python`
**Purpose:** Get the venv created and activated successfully.

**Prompt:** "ok step 2 now please"
**Purpose:** Get step-by-step instructions for installing Ollama and pulling the chosen model.

---

## Benchmarking & Model Selection

**Prompt:** "how thorough should the benchmark script be?" (answered: a few different prompts, short/long, to compare)
**Purpose:** Decide the design of the latency benchmark script.

**Prompt:** Requested the benchmark script content, then pasted the 3B model's benchmark output
**Purpose:** Get a Python script measuring tokens/sec and time-to-first-token across prompt lengths, then interpret whether 3B was fast enough.

**Prompt:** Pasted the 1.5B model's benchmark output after being asked to compare against 3B
**Purpose:** Decide between the 3B and 1.5B model based on measured latency data.

**Prompt:** "can you write me this model-selection.md in simple english"
**Purpose:** Turn the benchmark comparison and decision into a documented, README-ready write-up.

---

## Domain Policy & Conversation Manager (Phase III)

**Prompt:** "you can generate a reasonable placeholder that i can edit later and lets move on to step 5"
**Purpose:** Generate placeholder gym data (classes, trainers, membership plans, policies) and the initial conversation manager code.

**Prompt:** Pasted CLI test output showing the bot gave medical advice for a pet injury
**Purpose:** Diagnose why the system prompt's rules weren't being followed; led to switching from raw-text prompting (`/api/generate`) to structured chat messages (`/api/chat`).

**Prompt:** Pasted CLI output showing the bot still answered unrelated off-topic questions (cooking, dog bite, fever)
**Purpose:** Diagnose why rule-based prompting alone wasn't enough; led to designing a separate topic-classification layer.

**Prompt:** Pasted classifier test results showing "what are the bookings" was misclassified as off-topic
**Purpose:** Fix a classifier false-positive by broadening its few-shot examples.

**Prompt:** Pasted a full CLI test transcript, including a fabricated response with wrong trainer names and freeze fees, later clarified as the user's own manual edits, not model hallucination
**Purpose:** Investigate a suspected hallucination issue (ultimately a false alarm at the time).

---

## Backend & WebSocket (Phase IV)

**Prompt:** Asked whether session state should be in-memory-per-connection or session-ID-based with reconnect support (answered: session-ID based)
**Purpose:** Decide the session architecture before writing the FastAPI WebSocket server.

**Prompt:** "PS C:\...> python test_ws_client.py / Python was not found..."
**Purpose:** Fix a venv-not-activated issue in a second terminal window.

**Prompt:** Pasted a WebSocket handshake timeout traceback when testing 5 concurrent connections
**Purpose:** Diagnose why concurrent connections failed; led to discovering that synchronous `requests` calls were blocking FastAPI's async event loop, and replacing them with `httpx.AsyncClient`.

**Prompt:** Pasted a server traceback (`RuntimeError: Cannot call "send" once a close message has been sent`) after a mid-stream disconnect test
**Purpose:** Diagnose and fix improper `WebSocketDisconnect` handling in the WebSocket route.

---

## Frontend (Phase V)

**Prompt:** "everything is fine. what next?" then "i feel like my frontend is like not done at all. how can i improve it"
**Purpose:** Get a functional and visual review of the initial frontend, and a revised version with a typing indicator, avatars, timestamps, auto-focus, and an empty state.

**Prompt:** Screenshot showing "what was my last prompt" being incorrectly redirected as off-topic
**Purpose:** Diagnose a classifier context-awareness gap.

**Prompt:** "does this mean Maintain dialogue history for the session. This requirment is satisfied? or is it something else?" / "Stay faithful to earlier context across multiple turns. Is this satisfied too?"
**Purpose:** Verify, with concrete tests rather than assumption, which Phase III requirements were genuinely met.

**Prompt:** Screenshot showing "My name is Ahmed khan" incorrectly triggering a recall response
**Purpose:** Diagnose LLM-based recall detection as unreliable; led to replacing it with a deterministic Python keyword check.

**Prompt:** "my requirements.txt just turned chinese?" (pasted garbled text)
**Purpose:** Diagnose a Windows PowerShell UTF-16 encoding issue when redirecting `pip freeze` output.

---

## Phase VI: Production Readiness (First Pass)

**Prompt:** Pasted concurrent session test results showing sequential-looking completion times
**Purpose:** Determine whether concurrency was genuinely working or just appeared to be serialized due to Ollama's single-model inference constraint.

**Prompt:** Screenshot showing "ok bye" misclassified as off-topic
**Purpose:** Diagnose the missing "Closing" stage; led to adding a deterministic closing-keyword check.

**Prompt:** Screenshot showing "give me the whole membership plan not just basic" misclassified as off-topic
**Purpose:** Recognize a recurring structural pattern of classifier failures; led to designing the deterministic gym-keyword ON_TOPIC shortcut, reducing reliance on the LLM classifier to only genuinely ambiguous cases.

**Prompt:** "In this conversation history visible in UI. are we doing this?"
**Purpose:** Verify the Phase V requirement against an edge case (page refresh mid-session); led to adding session history restoration on WebSocket reconnect.

**Prompt:** "Are we doing all of these?" (re: Phase VI latency/correctness/failure-handling requirements)
**Purpose:** Identify that latency had only been benchmarked against raw Ollama, not the full pipeline; led to writing `test_latency_e2e.py` and measuring real end-to-end latency across all routing paths.

---

## Documentation (First Pass)

**Prompt:** "Ok my README is fully empty. help me write it fully and anything else that i have to write?"
**Purpose:** Generate the full README covering setup, architecture, business use case, conversation flow, example dialogues, context memory scheme, model selection, and known limitations.

**Prompt:** "give me the whole content for the README file here in text please" / "give me to me again as a file not text"
**Purpose:** Get the README delivered as an actual downloadable file rather than inline chat text.

**Prompt:** "can you tell me which part to update exactly for the readme?"
**Purpose:** Get precise, targeted edits to the README reflecting the recall/closing fixes without regenerating the whole document.

**Prompt:** "Give me fully updated Readme and do not add architecture diagrma in there i will make it myself"
**Purpose:** Get the final consolidated README with all fixes and end-to-end latency data included, while leaving the architecture diagram section as a placeholder for manual creation.

**Prompt:** "i have not made docs/ai-prompt-log.md Can you give me full content of it right now please"
**Purpose:** Generate this log itself, reconstructing the development history into a viva-ready record.

---

## Phase VI: Production Readiness (Second Pass — Peer Testing)

**Prompt:** Uploaded a friend's independent test report, listing three severity tiers of bugs found by someone other than the developer: (1) Critical — schedule hallucinations, including the model fabricating a non-existent Friday HIIT class and then "confirming" it as booked, mixing up Thursday's actual class times, and padding a trainer's availability with invented days; (2) High — four more messages misrouted as off-topic by the classifier ("my name is John Marston", "what is available on thursday evening", "ok i will take that one", "no that is all"); (3) Low — "i was bit by a dog" correctly redirected to the medical response, but only because the LLM classifier got it right, with no deterministic keyword backing it up. Asked: "Can you fix these as well and then give me updated AI prompt log"
**Purpose:** Fix all reported issues. This led to:
- A new deterministic "schedule-grounding guardrail" (`_find_invalid_day_class_pairs`, `_build_grounded_correction`) that checks the model's completed reply against the real `CLASS_SCHEDULE` data and appends a factual correction if it detects a fabricated (day, class) pairing, this is the first genuinely code-level (not just prompt-level) fix against hallucination in the project
- Four new grounding rules added to the system prompt (Rules 8-11), including the exact Friday-HIIT failure as a worked negative example
- Lowering the main model's generation temperature to reduce fabrication likelihood
- Expanding `GYM_KEYWORDS` to include day names, time-of-day words, and "available"/"availability"
- Adding a new `CONFIRMATION_KEYWORDS` list ("i will take", "sounds good", etc.) and `NAME_INTRO_KEYWORDS` list ("my name is", "call me") as additional deterministic ON_TOPIC routes
- Adding the missing "that is all" / "no that is all" variants to `CLOSING_KEYWORDS`
- Adding "bit by" to `MEDICAL_KEYWORDS`

**Prompt:** (Same message as above) — a second issue was caught independently during review of the fix, not something the friend's report flagged: the first draft of the grounding-guardrail fix buffered the entire model reply before sending it, which silently broke true token-by-token streaming, a direct regression against the Phase IV requirement. This was caught and corrected before being handed back: the guardrail now checks the reply only after it has already streamed live, and appends a correction afterward rather than replacing the whole reply.
**Purpose:** Preserve Phase IV's explicit streaming requirement while still adding the hallucination safety net.

**Prompt:** "can you give me fully updated ai prompt log now?"
**Purpose:** Update this log to reflect the peer-testing round of fixes.

---

## Note on Reproduction

All prompts above are reconstructed accurately from the actual chat history with Claude used during development. The assistant was used throughout as a pair-programming and debugging aid: writing initial code, diagnosing test failures and bugs from pasted output/screenshots/peer test reports, and explaining design tradeoffs. All code was reviewed, run, and tested locally before being accepted; the core prompt orchestration and conversation logic decisions (e.g. the switch from LLM-based to deterministic keyword-based routing for recall/closing/topic-shortcut detection, and the schedule-grounding guardrail) were made in direct response to real, observed test failures, not accepted blindly. The peer-testing round in particular is worth being able to speak to directly in the viva: an outside tester found real correctness bugs (including a genuinely serious one, a confirmed fake booking) that the developer's own testing had missed, and the fixes were reviewed carefully enough to catch and correct a second, unreported regression (broken streaming) introduced by the first draft of the fix itself.