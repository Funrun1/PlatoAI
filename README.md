# Platos

A voice assistant for Raspberry Pi (or any computer) using Claude as the brain.

Mic -> wake word -> speech-to-text -> Claude -> text-to-speech -> speaker

## Setup

1. Create a virtualenv and install:
   ```
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   python -c "import openwakeword; openwakeword.utils.download_models()"
   ```
   On a Pi first run: `sudo apt install -y portaudio19-dev libsndfile1 alsa-utils`

2. Add your API key:
   ```
   cp .env.example .env
   ```
   Then edit `.env` and paste your Anthropic API key.

3. Download a Piper voice (the `.onnx` and `.onnx.json` files) into `voices/`.
   Voices: https://huggingface.co/rhasspy/piper-voices

4. Wake word: train a "platos" model with the openWakeWord Colab notebook and save it
   as `models/platos.onnx`. Until then the script falls back to "hey_jarvis".

5. Run:
   ```
   python assistant.py
   ```

## Running on boot (Raspberry Pi)

Edit `platos.service` (replace `yourname`), then:
```
sudo cp platos.service /etc/systemd/system/
sudo systemctl enable --now platos
journalctl -u platos -f
```

## Tuning

If it misses you or false-triggers, change `WAKE_THRESHOLD` in `.env`.
If it cuts you off or never stops recording, change `SILENCE_RMS`.
If audio uses the wrong device, set `INPUT_DEVICE` / `OUTPUT_DEVICE`
(list them with `python -c "import sounddevice; print(sounddevice.query_devices())"`).
