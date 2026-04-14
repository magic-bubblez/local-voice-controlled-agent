import os
import subprocess
import ollama

MODEL = "qwen2.5-coder:3b"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def handle_filesystem(params: dict, context: dict) -> dict:
    """Create a file or folder in the output/ directory."""
    filename = params.get("filename", "untitled.txt")
    filetype = params.get("filetype", "file")

    path = os.path.join(OUTPUT_DIR, filename)

    # Safety: ensure we never write outside output/
    real_path = os.path.realpath(path)
    if not real_path.startswith(os.path.realpath(OUTPUT_DIR)):
        return {"status": "error", "message": "Cannot write outside output/ directory"}

    if filetype == "folder":
        os.makedirs(path, exist_ok=True)
        # Reveal the new folder in Finder
        try:
            subprocess.run(["open", path], timeout=5)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
        return {"status": "created", "type": "folder", "path": path}
    else:
        # If there's generated code in the context, use it as content
        content = context.get("generated_content", "")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) != OUTPUT_DIR else OUTPUT_DIR, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        # Auto-open in VS Code so user can see the file immediately
        try:
            subprocess.run(["open", "-a", "Visual Studio Code", path], timeout=5)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass  # VS Code not installed or failed — not critical
        return {"status": "created", "type": "file", "path": path, "size": len(content)}


def handle_llm_generate(params: dict, context: dict) -> dict:
    """Use the LLM to generate code, summaries, or chat responses."""
    action_type = context.get("current_action_type", "general_chat")

    if action_type == "write_code":
        language = params.get("language", "python")
        description = params.get("description", "")
        prompt = f"Write {language} code for: {description}. Return ONLY the code, no explanation, no markdown fences."

    elif action_type == "summarize":
        topic = params.get("topic", "")
        file_ref = params.get("file", "")

        # If user referenced a file, read its content and summarize that
        if file_ref:
            file_path = os.path.expanduser(file_ref) if file_ref else ""
            if not os.path.exists(file_path):
                file_path = _find_file(file_ref)

            if file_path and os.path.exists(file_path):
                content = _read_file_content(file_path)
                if content:
                    # Truncate very long content to fit context window
                    max_chars = 12000
                    if len(content) > max_chars:
                        content = content[:max_chars] + "\n\n[... content truncated ...]"
                    prompt = f"Summarize the following document clearly and concisely. Focus on the main points, key concepts, and important takeaways.\n\nDocument: {file_path}\n\nContent:\n{content}\n\nSummary:"
                else:
                    return {"status": "error", "type": action_type,
                            "content": f"Could not read content from file: {file_path}"}
            else:
                return {"status": "error", "type": action_type,
                        "content": f"Could not find file matching: {file_ref}"}
        else:
            prompt = f"Provide a clear, concise summary of: {topic}"

    else:  # general_chat
        query = params.get("query", "")
        prompt = query

    response = ollama.chat(model=MODEL, messages=[
        {"role": "user", "content": prompt}
    ])

    result_text = response["message"]["content"]

    # Strip markdown code fences if present
    if result_text.startswith("```"):
        lines = result_text.split("\n")
        # Remove first line (```python) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        result_text = "\n".join(lines).strip()

    # Store generated content in context so filesystem handler can use it
    if action_type == "write_code":
        context["generated_content"] = result_text

    # Speak the response for chat and summary (not for code — reading code aloud is awful)
    if action_type in ("general_chat", "summarize"):
        _speak(result_text)

    return {"status": "success", "type": action_type, "content": result_text}


def _speak(text: str):
    """Use macOS `say` to speak text aloud. Runs async so UI isn't blocked."""
    if not text:
        return
    # Truncate extremely long text to avoid endless speech
    if len(text) > 1500:
        text = text[:1500] + "... and more."
    try:
        # Popen = fire and forget, doesn't block the request
        subprocess.Popen(["say", text])
    except Exception:
        pass


def _read_file_content(path: str) -> str:
    """Read the content of a file. Handles PDFs and plain text."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            return ""

    # Plain text / code / markdown / etc.
    if ext in (".txt", ".md", ".py", ".js", ".go", ".html", ".css", ".json", ".yaml", ".yml", ".sh", ".rs", ".c", ".cpp", ".java", ""):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

    return ""


def _find_file(query: str) -> str:
    """Search common locations for a file matching the query."""
    import glob

    # Locations to search, in priority order
    search_dirs = [
        os.path.expanduser("~/Desktop"),
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Documents"),
        os.path.expanduser("~"),
    ]

    # Clean up the query — extract likely filename keywords
    query_lower = query.lower().strip()

    home = os.path.expanduser("~")

    # Use mdfind with filename search
    try:
        result = subprocess.run(
            ["mdfind", "-onlyin", home, f"kMDItemFSName == '*{query_lower}*'c"],
            capture_output=True, text=True, timeout=5
        )
        matches = [l for l in result.stdout.strip().split("\n") if l]
        if matches:
            return matches[0]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: broad content search, filtered to common file types
    try:
        result = subprocess.run(
            ["mdfind", "-onlyin", home, query_lower],
            capture_output=True, text=True, timeout=5
        )
        valid_ext = ('.pdf', '.png', '.jpg', '.jpeg', '.txt', '.doc', '.docx', '.mp3', '.mp4', '.mov')
        matches = [l for l in result.stdout.strip().split("\n") if l and l.lower().endswith(valid_ext)]
        # Score by how many query words appear in the path
        query_words = query_lower.split()
        scored = []
        for m in matches:
            score = sum(1 for w in query_words if w in m.lower())
            scored.append((score, m))
        scored.sort(reverse=True)
        if scored and scored[0][0] > 0:
            return scored[0][1]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return ""


APP_ALIASES = {
    "chrome": "Google Chrome",
    "vscode": "Visual Studio Code",
    "vs code": "Visual Studio Code",
    "code": "Visual Studio Code",
    "firefox": "Firefox",
    "safari": "Safari",
    "terminal": "Terminal",
    "iterm": "iTerm",
    "notes": "Notes",
    "music": "Music",
    "spotify": "Spotify",
    "slack": "Slack",
    "discord": "Discord",
    "finder": "Finder",
    "preview": "Preview",
    "pages": "Pages",
    "numbers": "Numbers",
    "keynote": "Keynote",
    "xcode": "Xcode",
    "postman": "Postman",
}


def handle_open_app(params: dict, context: dict) -> dict:
    """Open a macOS application, file, or URL."""
    app_name = params.get("app_name", "")
    file_path = params.get("file", "")
    url = params.get("url", "")

    # Guard: if nothing specified, return error instead of running bare `open`
    if not app_name and not file_path and not url:
        return {
            "status": "error",
            "message": "Could not determine what to open — no app, file, or URL was specified."
        }

    # Resolve common app name aliases
    if app_name:
        app_name = APP_ALIASES.get(app_name.lower(), app_name)

    # Expand ~ in file paths
    if file_path:
        file_path = os.path.expanduser(file_path)

    # If file_path looks like a vague reference (no extension, no /), search for it
    if file_path and not os.path.exists(file_path):
        found = _find_file(file_path)
        if found:
            file_path = found

    cmd = ["open"]
    if app_name:
        cmd.extend(["-a", app_name])
    if url:
        cmd.append(url)
    elif file_path:
        cmd.append(file_path)

    try:
        subprocess.run(cmd, check=True, timeout=5)
        target = url or file_path or app_name
        return {"status": "opened", "app": app_name, "target": target}
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return {"status": "error", "message": str(e)}


def handle_screenshot(params: dict, context: dict) -> dict:
    """Take a screenshot using macOS screencapture."""
    import datetime

    mode = params.get("mode", "full")  # "full", "window", "selection"
    filename = params.get("filename", "")

    if not filename:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{ts}.png"

    if not filename.endswith(".png"):
        filename += ".png"

    path = os.path.join(OUTPUT_DIR, filename)

    # Safety: ensure path stays inside output/
    real_path = os.path.realpath(path)
    if not real_path.startswith(os.path.realpath(OUTPUT_DIR)):
        return {"status": "error", "message": "Cannot save outside output/ directory"}

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # screencapture flags:
    #   (no flags)  = full screen
    #   -w          = window (user clicks a window)
    #   -s          = selection (user drags a region)
    cmd = ["screencapture"]
    if mode == "window":
        cmd.append("-w")
    elif mode == "selection":
        cmd.append("-s")
    cmd.append(path)

    try:
        subprocess.run(cmd, check=True, timeout=30)
        if os.path.exists(path):
            # Open the screenshot in Preview so user sees the result
            try:
                subprocess.run(["open", "-a", "Preview", path], timeout=5)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass
            return {"status": "captured", "path": path, "mode": mode}
        else:
            return {"status": "error", "message": "Screenshot was cancelled"}
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return {"status": "error", "message": str(e)}
