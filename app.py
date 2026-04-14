import json
import logging
import os
import time
import gradio as gr
from stt import transcribe
from classifier import classify_intent
from orchestrator import execute

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("voice-agent")

# Session history — persists across commands within one browser session
session_history = []


def process_audio(audio_path):
    """Full pipeline: audio -> text -> intent -> execution -> display."""
    if audio_path is None:
        return "No audio provided.", "", "", "", "", format_history()

    t0 = time.time()

    # Layer 2: Speech-to-Text
    log.info("STT: transcribing %s", audio_path)
    t_stt_start = time.time()
    transcribed_text = transcribe(audio_path)
    t_stt = time.time() - t_stt_start
    log.info("STT result (%.2fs): '%s'", t_stt, transcribed_text)
    if not transcribed_text.strip():
        return "Could not transcribe audio.", "", "", "", "", format_history()

    # Layer 3: Intent Classification
    log.info("Classifier: analyzing intent...")
    t_cls_start = time.time()
    intent = classify_intent(transcribed_text, session_history)
    t_cls = time.time() - t_cls_start
    actions = intent.get("actions", [])
    log.info("Classifier result (%.2fs): %s", t_cls, [a.get("type") for a in actions])
    intent_display = json.dumps(intent, indent=2)

    # Layer 4: Orchestration + Tool Execution
    log.info("Orchestrator: executing %d action(s)...", len(actions))
    t_orch_start = time.time()
    context = execute(actions)
    t_orch = time.time() - t_orch_start
    results = context.get("results", [])
    for r in results:
        log.info("  [%s] -> %s", r.get("action"), r.get("status"))

    t_total = time.time() - t0
    timing = (
        f"⏱️  Total: {t_total:.2f}s\n"
        f"   • STT:        {t_stt:.2f}s\n"
        f"   • Classifier: {t_cls:.2f}s\n"
        f"   • Orchestrator + tools: {t_orch:.2f}s"
    )

    action_summary = "\n".join(f"[{r['action']}] -> {r['status']}" for r in results)
    final_output = build_output(results)

    # Save to session history
    session_history.append({
        "raw_text": transcribed_text,
        "actions_result": intent,
    })

    return (
        transcribed_text,
        intent_display,
        action_summary,
        final_output,
        timing,
        format_history(),
    )


def process_text(text):
    """Same pipeline but skip STT — for typed input."""
    if not text or not text.strip():
        return "No text provided.", "", "", "", "", format_history()

    t0 = time.time()

    log.info("Text input: '%s'", text)
    log.info("Classifier: analyzing intent...")
    t_cls_start = time.time()
    intent = classify_intent(text, session_history)
    t_cls = time.time() - t_cls_start
    actions = intent.get("actions", [])
    log.info("Classifier result (%.2fs): %s", t_cls, [a.get("type") for a in actions])
    intent_display = json.dumps(intent, indent=2)

    log.info("Orchestrator: executing %d action(s)...", len(actions))
    t_orch_start = time.time()
    context = execute(actions)
    t_orch = time.time() - t_orch_start
    results = context.get("results", [])
    for r in results:
        log.info("  [%s] -> %s", r.get("action"), r.get("status"))

    t_total = time.time() - t0
    timing = (
        f"⏱️  Total: {t_total:.2f}s\n"
        f"   • Classifier: {t_cls:.2f}s\n"
        f"   • Orchestrator + tools: {t_orch:.2f}s"
    )

    action_summary = "\n".join(f"[{r['action']}] -> {r['status']}" for r in results)
    final_output = build_output(results)

    session_history.append({
        "raw_text": text,
        "actions_result": intent,
    })

    return (
        text,
        intent_display,
        action_summary,
        final_output,
        timing,
        format_history(),
    )


def build_output(results):
    """Build a human-readable final output string from handler results."""
    parts = []
    for r in results:
        action = r.get("action", "unknown")
        status = r.get("status", "")

        if status == "error":
            parts.append(f"❌ [{action}] {r.get('message', r.get('content', 'unknown error'))}")
            continue

        if action == "write_code" or action == "summarize" or action == "general_chat":
            content = r.get("content", "")
            parts.append(f"📝 [{action}]\n{content}")

        elif action == "create_file":
            path = r.get("path", "")
            ftype = r.get("type", "file")
            if ftype == "folder":
                parts.append(f"📁 Folder created: {path}")
            else:
                # Show file path AND preview of contents
                preview = ""
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        preview = f.read()
                except Exception:
                    preview = "(binary or unreadable file)"
                if len(preview) > 2000:
                    preview = preview[:2000] + "\n... [truncated]"
                parts.append(f"📄 File created: {path}\n\n--- content ---\n{preview}")

        elif action == "open_app":
            target = r.get("target", "") or r.get("app", "")
            parts.append(f"🚀 Opened: {target}")

        elif action == "screenshot":
            path = r.get("path", "")
            parts.append(f"📸 Screenshot saved: {path}")

        else:
            parts.append(f"[{action}] {status}")

    return "\n\n".join(parts)


def format_history():
    """Format session history for display."""
    if not session_history:
        return "No commands yet."

    lines = []
    for i, entry in enumerate(session_history, 1):
        actions = [a["type"] for a in entry["actions_result"].get("actions", [])]
        lines.append(f"{i}. \"{entry['raw_text']}\" -> [{', '.join(actions)}]")

    return "\n".join(lines)


def clear_history():
    """Reset session history."""
    session_history.clear()
    return "No commands yet."


# ---- Gradio UI ----

with gr.Blocks(title="Voice Agent", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Voice-Controlled Local AI Agent")
    gr.Markdown("Speak a command or type it. The agent will classify your intent and execute it locally.")

    with gr.Row():
        # Left column: inputs
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Speak or upload audio",
            )
            audio_btn = gr.Button("Process Audio", variant="primary")

            gr.Markdown("---")
            text_input = gr.Textbox(
                label="Or type a command",
                placeholder="e.g., Create a Python file with a retry function",
            )
            text_btn = gr.Button("Process Text", variant="secondary")

        # Right column: outputs
        with gr.Column(scale=2):
            transcription_out = gr.Textbox(label="Transcribed Text", interactive=False)
            intent_out = gr.Textbox(label="Detected Intent (JSON)", interactive=False, lines=6)
            action_out = gr.Textbox(label="Actions Taken", interactive=False)
            result_out = gr.Textbox(label="Final Output", interactive=False, lines=20, max_lines=40)
            timing_out = gr.Textbox(label="Pipeline Timing", interactive=False, lines=5)

    gr.Markdown("---")
    with gr.Row():
        history_out = gr.Textbox(
            label="Session History",
            interactive=False,
            lines=6,
            value="No commands yet.",
        )
        clear_btn = gr.Button("Clear History")

    # Wire up the buttons
    outputs = [transcription_out, intent_out, action_out, result_out, timing_out, history_out]

    audio_btn.click(fn=process_audio, inputs=[audio_input], outputs=outputs)
    text_btn.click(fn=process_text, inputs=[text_input], outputs=outputs)
    clear_btn.click(fn=clear_history, inputs=[], outputs=[history_out])


if __name__ == "__main__":
    app.launch()
