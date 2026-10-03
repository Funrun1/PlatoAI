# Platos

A voice assistant for Raspberry Pi (or any computer) using Claude as the brain.
Made by Aqeel. AI helped a bit 

Mic -> wake word -> speech-to-text -> Claude -> text-to-speech -> speaker

## Setup

1. Create a virtualenv and install:
```
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   python -c "import openwakeword; openwakeword.utils.download_models()"
```
   On a Pi, first run: `sudo apt install -y portaudio19-dev libsndfile1 alsa-utils`

2. Make your `.env` file and add your API key:
```