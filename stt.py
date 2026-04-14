from faster_whisper import WhisperModel


# Load once, reuse across requests. "base" model, running on CPU.
# On M1 Mac, "auto" compute type lets CTranslate2 pick the best option.
model = WhisperModel("base", device="cpu", compute_type="auto")


def transcribe(audio_path: str) -> str:
    """Takes a filepath to an audio file, returns the transcribed text."""
    segments, info = model.transcribe(audio_path)
    text = " ".join(segment.text.strip() for segment in segments)
    return text
