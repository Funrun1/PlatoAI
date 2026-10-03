"""
Platos: a Raspberry Pi voice assistant backed by Claude.

Pipeline:  mic -> openWakeWord ("Platos") -> faster-whisper (STT)
           -> Claude API -> Piper (TTS) -> speaker

Works on Mac, Windows, Linux and Raspberry Pi. Settings come from .env.
"""
import os
import subprocess
import sys
import tempfile
import wave

import anthropic
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openwakeword.model import Model

load_dotenv()

# ---------- Config (override any of these in .env) ----------
SAMPLE_RATE = 16000
FRAME = 1280  # 80 ms of audio, what openWakeWord expects
WAKE_MODEL_PATH = os.getenv("WAKE_MODEL_PATH", "models/platos.onnx")
FALLBACK_WAKE_WORD = "hey_jarvis"  # used until your Platos model exists
WAKE_THRESHOLD = float(os.getenv("WAKE_THRESHOLD", "0.5"))
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
PIPER_VOICE = os.getenv("PIPER_VOICE", "voices/en_US-lessac-medium.onnx")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")  # try tiny.en on a Pi 4
SILENCE_RMS = float(os.getenv("SILENCE_RMS", "500"))
SILENCE_SECONDS = float(os.getenv("SILENCE_SECONDS", "1.0"))
MAX_RECORD_SECONDS = int(os.getenv("MAX_RECORD_SECONDS", "10"))
INPUT_DEVICE = os.getenv("INPUT_DEVICE")    # optional device index or name
OUTPUT_DEVICE = os.getenv("OUTPUT_DEVICE")  # optional device index or name

SYSTEM_PROMPT = (
    "You are Platos, a voice assistant running on a Raspberry Pi. "
    "Your replies are spoken aloud, so keep them short (1-3 sentences), "
    "conversational, and free of markdown, lists, or emoji."
)


def _device(value):
    if value is None or value == "":
        return None
    return int(value) if value.isdigit() else value


# ---------- Setup ----------
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

if os.path.exists(WAKE_MODEL_PATH):
    wake = Model(wakeword_models=[WAKE_MODEL_PATH], inference_framework="onnx")
else:
    print(f"[warn] {WAKE_MODEL_PATH} not found, using '{FALLBACK_WAKE_WORD}' for now.")
    wake = Model(wakeword_models=[FALLBACK_WAKE_WORD], inference_framework="onnx")

stt = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
history = []  # rolling conversation memory


def speak(text: str) -> None:
    """Synthesize with Piper and play through sounddevice (cross-platform)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "reply.wav")
        subprocess.run(
            [sys.executable, "-m", "piper", "--model", PIPER_VOICE, "--output_file", out],
            input=text.encode(),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with wave.open(out, "rb") as wf:
            rate, channels = wf.getframerate(), wf.getnchannels()
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        if channels > 1:
            audio = audio.reshape(-1, channels)
        sd.play(audio, samplerate=rate, device=_device(OUTPUT_DEVICE))
        sd.wait()


def record_command(stream) -> np.ndarray:
    """Record after the wake word until the user stops talking."""
    frames, silent_frames, heard_speech = [], 0, False
    frames_per_sec = SAMPLE_RATE / FRAME
    for _ in range(int(MAX_RECORD_SECONDS * frames_per_sec)):
        data, _ = stream.read(FRAME)
        chunk = data[:, 0]
        frames.append(chunk)
        rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
        if rms > SILENCE_RMS:
            heard_speech, silent_frames = True, 0
        else:
            silent_frames += 1
        if heard_speech and silent_frames > SILENCE_SECONDS * frames_per_sec:
            break
    return np.concatenate(frames)


def transcribe(audio_int16: np.ndarray) -> str:
    audio = audio_int16.astype(np.float32) / 32768.0
    segments, _ = stt.transcribe(audio, language="en", vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


def ask_claude(user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    del history[:-12]  # keep the last 12 messages
    while history and history[0]["role"] != "user":
        history.pop(0)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=history,
    )
    reply = "".join(b.text for b in response.content if b.type == "text")
    history.append({"role": "assistant", "content": reply})
    return reply


def main() -> None:
    print("Listening for the wake word... (Ctrl+C to quit)")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=FRAME,
        device=_device(INPUT_DEVICE),
    ) as stream:
        while True:
            data, _ = stream.read(FRAME)
            scores = wake.predict(data[:, 0])
            if max(scores.values()) < WAKE_THRESHOLD:
                continue

            wake.reset()
            print("Wake word detected")
            audio = record_command(stream)
            text = transcribe(audio)
            if not text:
                continue
            print(f"You: {text}")
            try:
                reply = ask_claude(text)
            except anthropic.APIError as e:
                print(f"Claude error: {e}")
                reply = "Sorry, I couldn't reach my brain just now."
            print(f"Platos: {reply}")
            speak(reply)
            wake.reset()


if __name__ == "__main__":
    main()
