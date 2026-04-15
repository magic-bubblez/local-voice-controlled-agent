# Voice Agent

A local voice-controlled AI agent that runs entirely on your Mac. Say a command — it transcribes, understands intent, and executes tools on your machine. No cloud, no APIs, no internet (unless you explicitly ask it to search the web).

## What It Does

| You say | It does |
|---|---|
| "Create a folder called experiments" | Makes the folder, reveals it in Finder |
| "Write a Go function to check if a string is a palindrome and save it" | Generates the code, saves to `output/palindrome.go`, opens in VS Code |
| "Summarize the DDIA pdf" | Finds the file via Spotlight, extracts text with pypdf, summarizes |
| "Open github.com and take a screenshot" | Opens browser, waits for render, captures screen, opens in Preview |
| "What is the difference between TCP and UDP?" | Local LLM chat response |

## Architecture

Five layers, clean interfaces, all local:

```
[Browser mic/upload] → audio bytes
    → [faster-whisper base] → text string
        → [qwen2.5-coder:3b via Ollama] → JSON action list
            → [Orchestrator: registry + exec context] → results
                → [Gradio UI] → displays everything
```

| Layer | File | Job |
|---|---|---|
| Audio input | `app.py` (Gradio component) | Capture from mic / upload |
| Speech-to-text | `stt.py` | Transcribe audio locally |
| Intent classifier | `classifier.py` | Text → structured JSON plan |
| Orchestrator | `orchestrator.py` | Sequence actions, dispatch to handlers |
| Tool handlers | `tools.py` | Actually execute things on the machine |
| UI | `app.py` | Display transcription, intent, actions, results |

The classifier uses an LLM to understand what you want. The orchestrator is a pure dictionary lookup — no intelligence, just a registry mapping action types to handler functions. Adding a new capability means: one prompt example, one handler function, one line in the registry. Nothing else changes.

## Capabilities

- **create_file** — Create files or folders in `output/` (sandboxed)
- **write_code** — Generate code via the LLM, save to file
- **summarize** — Summarize a concept or a local file (PDF/text)
- **general_chat** — Conversational LLM response
- **open_app** — Launch macOS apps, open local files (via Spotlight search), open URLs
- **screenshot** — Full screen / window pick / drag selection, auto-opens in Preview

Compound commands work — "write a function and save it" emits `[write_code, create_file]` and the orchestrator sequences them.

## Setup

Requires: macOS (uses `open`, `screencapture`, `mdfind`), Python 3.11+, [Ollama](https://ollama.com).

```bash
# 1. Clone
git clone <repo> voice-agent && cd voice-agent

# 2. Virtualenv
python3.12 -m venv venv
source venv/bin/activate

# 3. Install deps
pip install faster-whisper gradio ollama pypdf

# 4. Install Ollama model
ollama pull qwen2.5-coder:3b

# 5. Run
python app.py
# Opens on http://localhost:7860
```

First run will also download the faster-whisper base model (~150 MB) to `~/.cache/huggingface/`.

## Design Decisions

- **3B model over 7B**: Benchmarked both (`compare_models.py`). 7B was 100% accurate vs 93% for 3B, but on 16 GB RAM the 7B caused swap pressure and responses took 3–4× longer. 3B is the demo-friendly tradeoff.
- **faster-whisper over HuggingFace Transformers**: Same Whisper model, no PyTorch dependency, ~1.5 GB RAM saved. CTranslate2 backend is 2–4× faster.
- **Gradio over Streamlit**: Pipeline-shaped system maps cleanly to Gradio's function-wiring model.
- **Flat action list, not dependency graph**: LLMs produce JSON reliably; the orchestrator owns execution order via hardcoded `ORDERING_RULES`.
- **Session memory as a Python list**: Appended per command, sent to the LLM on the next call (replay pattern). Capped at the last 3 entries to prevent bad classifications from poisoning future ones.
- **Sandbox**: All file operations are restricted to `output/` with realpath checks.

## Hardware Constraints

Built on an M1 MacBook Pro, 16 GB RAM. The 7B model technically runs but triggers swap; 3B is comfortable alongside faster-whisper + Gradio + browser tabs.

## Project Layout

```
voice-agent/
├── app.py              # Gradio UI + pipeline wiring + timing instrumentation
├── stt.py              # faster-whisper transcription
├── classifier.py       # Ollama-based intent classification + prompt
├── orchestrator.py     # Registry pattern + execution context + ordering
├── tools.py            # Five handler functions
├── compare_models.py   # Benchmark script (3B vs 7B)
├── requirements.txt
└── output/             # Sandboxed file operations
```

## Limitations / Future Work

- No cross-session memory — when the server restarts, history is gone. A real personal agent would need persistent memory (what Mem0 does as a product).
- The 3B model struggles with nuanced phrasings (e.g. "set up a new Go project" — 3B reads "set up" as "open"). Patched with a post-processing corrector for the most common cases.
- Async side effects (browser render time) are handled with a hardcoded 3 s sleep when `screenshot` follows `open_app` — a cleaner design would use event signals.
- No confirmation step before destructive actions.
