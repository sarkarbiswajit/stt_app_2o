# How This App Works

This Streamlit app is built to detect language from audio, transcribe speech to text, and translate the transcript to English.

## Main Components

- `app.py`
  - Uses Streamlit to create a web interface.
  - Provides two input methods:
    - **Live recording** using the `audio-recorder-streamlit` component.
    - **Audio upload** for files such as WAV, MP3, M4A, OGG, and FLAC.
  - Sends audio to the Bhashini API for processing.

- `requirements.txt`
  - Lists Python dependencies for the app.
  - Includes:
    - `streamlit`
    - `requests`
    - `audio-recorder-streamlit`

- `.venv/`
  - Local Python virtual environment created inside the folder.
  - Keeps dependencies isolated from other projects.

## What Happens When You Use the App

1. You record audio directly in the browser or upload a file.
2. The app normalizes audio to the expected input shape:
   - WAV PCM
   - 16 kHz sample rate
   - mono channel
   - 16-bit depth
3. The normalized audio is converted into base64 and sent to the Bhashini endpoint:
   - `https://anuvaad-backend.bhashini.co.in/v1/pipeline`
4. The app first detects the language using `audio-lang-detection`.
5. It then transcribes the audio with the `asr` model.
6. Finally, it translates the resulting text to English using the `translation` model.

## Accepted Audio Input Types

- Recorded live audio from the browser (WebM/Opus captured by `audio-recorder-streamlit`)
- Uploaded files:
  - WAV
  - MP3
  - M4A
  - OGG
  - FLAC

## Audio Normalization Details

- **Live recording** is captured by the browser component and normalized to WAV PCM 16 kHz mono before sending.
- **Uploaded audio** is read by the app, resampled if needed, converted to mono, and exported as WAV PCM 16 kHz.
- This ensures the app sends a consistent audio shape to the STT API.

## What the App Sends to the API

- The app sends the normalized audio as a base64 string inside the `audio` field.
- For language detection, the payload uses `audio-lang-detection` with `serviceId` set to `bhashini/iitmandi/audio-lang-detection/gpu`.
- For transcription, the payload uses `asr` with `serviceId` set to `bhashini/ai4bharat/conformer-multilingual-asr`.
- For translation, the payload uses `translation` with `serviceId` set to `ai4bharat/indictrans-v2-all-gpu--t4`.

## Audio Processing Pipeline

1. Normalize audio format.
2. Encode audio to base64.
3. Call Bhashini language detection.
4. Call Bhashini ASR.
5. Call Bhashini translation.
6. Display detected language, transcript, and English translation.

## User Flow

- **Record Audio** tab:
  - Click the record button.
  - Stop recording when finished.
  - Click `Process Recording`.

- **Upload Audio** tab:
  - Choose an audio file.
  - Click `Process Audio`.

## Output

- The app displays:
  - Detected language
  - Transcript of the audio
  - English translation of the transcript

## How to Run

```bash
cd "c:\Users\Biswajit\Desktop\ai hackathon\detect_stt_app"
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The app requires internet access to call the Bhashini API.
- Uploaded and recorded audio is normalized to WAV PCM mono at 16 kHz.
- Errors from the API are shown inside the Streamlit app.
