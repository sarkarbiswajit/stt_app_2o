import streamlit as st
import base64
import requests
import io
from pydub import AudioSegment, effects, silence
from audio_recorder_streamlit import audio_recorder

# Set page config
st.set_page_config(
    page_title="STT Audio App",
    page_icon="🎤",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        padding: 0.5rem;
        font-size: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🎤 Audio Language Detection & STT")
st.markdown("Upload or record audio to detect language, transcribe speech, and translate to English.")

# API endpoint
URL = "https://anuvaad-backend.bhashini.co.in/v1/pipeline"

if "logs" not in st.session_state:
    st.session_state.logs = ["App started"]


def log(message):
    st.session_state.logs.append(message)


def render_log():
    with st.sidebar:
        st.title("Process Log")
        st.markdown("---")
        for item in st.session_state.logs:
            st.markdown(f"- {item}")


def get_audio_format(filename, mime_type=None):
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[1].lower()
        return ext
    if mime_type:
        if "wav" in mime_type:
            return "wav"
        if "mp3" in mime_type:
            return "mp3"
        if "ogg" in mime_type:
            return "ogg"
        if "flac" in mime_type:
            return "flac"
        if "m4a" in mime_type or "mp4" in mime_type:
            return "mp4"
    return "wav"


def normalize_audio(audio_bytes, source_format="wav"):
    try:
        audio_file = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_file(audio_file, format=source_format)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        output = io.BytesIO()
        audio.export(output, format="wav")
        return output.getvalue()
    except Exception as e:
        st.error(f"Audio normalization failed: {e}")
        log(f"Audio normalization failed: {e}")
        return None


def preprocess_audio(audio_bytes):
    try:
        log("Running Stage 1 preprocessing: silence trim, volume normalize, noise reduction")
        audio_file = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_file(audio_file, format="wav")

        # Basic noise reduction / hum removal
        audio = audio.high_pass_filter(100)

        # Trim leading/trailing silence
        nonsilent_ranges = silence.detect_nonsilent(audio, min_silence_len=300, silence_thresh=-50)
        if nonsilent_ranges:
            start_ms = nonsilent_ranges[0][0]
            end_ms = nonsilent_ranges[-1][1]
            audio = audio[start_ms:end_ms]
            log(f"Trimmed silence: {start_ms}ms -> {end_ms}ms")
        else:
            log("No silence detected for trimming")

        # Volume normalization
        audio = effects.normalize(audio)

        # Ensure final shape
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        output = io.BytesIO()
        audio.export(output, format="wav")
        log("Stage 1 preprocessing complete")
        return output.getvalue()
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        log(f"Preprocessing failed: {e}")
        return None


def get_audio_metadata(audio_bytes, source_format="wav"):
    try:
        log(f"Reading audio metadata for format: {source_format}")
        audio_file = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_file(audio_file, format=source_format)
        metadata = {
            "format": source_format,
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "sample_width_bits": audio.sample_width * 8,
            "duration_seconds": round(len(audio) / 1000.0, 2),
            "frame_count": int(audio.frame_count())
        }
        log(f"Audio metadata read: {metadata}")
        return metadata
    except Exception as e:
        st.error(f"Could not read audio metadata: {e}")
        log(f"Audio metadata read failed: {e}")
        return None

LANGUAGE_MAP = {
    "hi": "Hindi",
    "en": "English",
    "mr": "Marathi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
    "or": "Odia",
    "as": "Assamese",
    "sd": "Sindhi",
    "ne": "Nepali",
    "si": "Sinhala",
    "kok": "Konkani",
    "doi": "Dogri",
    "bho": "Bhojpuri",
    "mag": "Magahi",
    "raj": "Rajasthani",
    "mai": "Maithili",
    "lang/unknown": "Unknown"
}

def get_detected_lang(audio_base64):
    """Detect the audio language using Bhashini language detection."""
    try:
        log("Sending language detection request")
        payload = {
            "pipelineTasks": [{
                "taskType": "audio-lang-detection",
                "config": {
                    "serviceId": "bhashini/iitmandi/audio-lang-detection/gpu",
                    "audioFormat": "wav",
                    "samplingRate": 16000
                }
            }],
            "inputData": {
                "input": [{"source": ""}],
                "audio": [{"audioContent": audio_base64}]
            }
        }
        response = requests.post(URL, json=payload, timeout=30)
        if response.status_code == 200:
            log("Language detection succeeded")
            return response.json()["pipelineResponse"][0]["output"][0]["langPrediction"][0]["langCode"]
        else:
            st.error(f"Detection Error: {response.text}")
            log(f"Language detection failed: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error detecting language: {e}")
        log(f"Language detection exception: {e}")
        return None


def transcribe_audio(audio_base64, source_lang, use_preprocessors=False):
    """Transcribe audio to text using Bhashini ASR."""
    try:
        log("Sending ASR transcription request")
        config = {
            "language": {"sourceLanguage": source_lang},
            "serviceId": "bhashini/ai4bharat/conformer-multilingual-asr",
            "audioFormat": "wav",
            "samplingRate": 16000,
            "postProcessors": ["tn"]
        }
        if use_preprocessors:
            config["preProcessors"] = ["vad", "denoiser"]
            log("Including ASR preProcessors: vad, denoiser")

        payload = {
            "pipelineTasks": [
                {
                    "taskType": "asr",
                    "config": config
                }
            ],
            "inputData": {
                "input": [{"source": ""}],
                "audio": [{"audioContent": audio_base64}]
            }
        }
        response = requests.post(URL, json=payload, timeout=30)
        if response.status_code == 200:
            log("ASR transcription succeeded")
            return response.json()["pipelineResponse"][0]["output"][0]["source"]
        else:
            st.error(f"ASR Error: {response.text}")
            log(f"ASR transcription failed: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        log(f"ASR transcription exception: {e}")
        return None


def translate_to_english(text, source_lang):
    """Translate transcript to English using Bhashini translation."""
    try:
        log("Sending translation request")
        payload = {
            "pipelineTasks": [
                {
                    "taskType": "translation",
                    "config": {
                        "language": {
                            "sourceLanguage": source_lang,
                            "targetLanguage": "en"
                        },
                        "serviceId": "ai4bharat/indictrans-v2-all-gpu--t4",
                        "postProcessors": ["glossary"]
                    }
                }
            ],
            "inputData": {
                "input": [{"source": text}]
            }
        }
        response = requests.post(URL, json=payload, timeout=30)
        if response.status_code == 200:
            log("Translation succeeded")
            return response.json()["pipelineResponse"][0]["output"][0]["target"]
        else:
            st.error(f"Translation Error: {response.text}")
            log(f"Translation failed: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error translating text: {e}")
        log(f"Translation exception: {e}")
        return None


# Audio input tabs
tab_record, tab_upload = st.tabs(["🎙️ Record Audio", "📁 Upload Audio"])

with tab_record:
    st.subheader("Live Recording")
    st.markdown("Record audio directly from your microphone.")
    audio_bytes = audio_recorder(
        text="Click 🎙️ to record",
        recording_color="#e74c3c",
        neutral_color="#d3d3d3",
        sample_rate=16000
    )

    preprocess_record = st.checkbox("Enable Stage 1 preprocessing (API VAD + denoiser)", value=False, key="preprocess_record")

    if audio_bytes:
        st.success("✅ Recording captured!")
        st.audio(audio_bytes, format="audio/wav")
        log("Recording captured from microphone")

        original_meta = get_audio_metadata(audio_bytes, source_format="wav")
        if original_meta is not None:
            with st.expander("Original recording metadata"):
                st.write(original_meta)

        if st.button("🚀 Process Recording", use_container_width=True, key="record_button"):
            log("Process Recording button clicked")
            log("Normalizing recorded audio to WAV PCM 16kHz mono")
            normalized_bytes = normalize_audio(audio_bytes, source_format="wav")
            if normalized_bytes is None:
                st.error("Could not normalize recorded audio.")
                log("Recorded audio normalization failed")
            else:
                log("Recorded audio normalized successfully")
                normalized_meta = get_audio_metadata(normalized_bytes, source_format="wav")

                with st.expander("Normalized payload metadata"):
                    st.write({
                        "format": "wav",
                        "sample_rate": 16000,
                        "channels": 1,
                        "sample_width_bits": 16,
                        "encoded_as": "PCM"
                    })
                if normalized_meta is not None:
                    with st.expander("Post-normalization metadata"):
                        st.write(normalized_meta)

                processed_bytes = normalized_bytes
                if preprocess_record:
                    log("Stage 1 preprocessing enabled")
                    preprocessed_bytes = preprocess_audio(normalized_bytes)
                    if preprocessed_bytes is not None:
                        processed_bytes = preprocessed_bytes
                        with st.expander("Preprocessed audio preview"):
                            st.audio(processed_bytes, format="audio/wav")
                        preprocessed_meta = get_audio_metadata(processed_bytes, source_format="wav")
                        if preprocessed_meta is not None:
                            with st.expander("Preprocessed payload metadata"):
                                st.write(preprocessed_meta)
                    else:
                        st.warning("Preprocessing failed, continuing with normalized audio")
                        log("Local preprocessing failed, continuing without it")
                else:
                    log("Stage 1 preprocessing disabled for recording")

                audio_base64 = base64.b64encode(processed_bytes).decode("utf-8")
                log("Sending normalized audio to language detection")
                with st.spinner("🔍 Detecting language..."):
                    detected_lang = get_detected_lang(audio_base64)

                if detected_lang:
                    language_name = LANGUAGE_MAP.get(detected_lang.lower(), detected_lang.upper())
                    log(f"Detected language: {language_name}")
                    st.success(f"Detected Language: {language_name}")
                    log("Sending audio to ASR for transcription")
                    with st.spinner("📝 Transcribing audio..."):
                        transcript = transcribe_audio(audio_base64, detected_lang, use_preprocessors=preprocess_record)

                    if transcript:
                        log("Received transcript from ASR")
                        st.info("**Transcript:**")
                        st.write(transcript)

                        log("Sending transcript to translation")
                        with st.spinner("🌐 Translating to English..."):
                            english_translation = translate_to_english(transcript, detected_lang)

                        if english_translation:
                            log("Received English translation")
                            st.info("**English Translation:**")
                            st.write(english_translation)
                        else:
                            log("Translation failed")
                    else:
                        st.error("Failed to transcribe audio.")
                        log("ASR transcription failed")
                else:
                    st.error("Language detection failed")
                    log("Language detection failed")


with tab_upload:
    st.subheader("Upload Your Audio File")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])

    if uploaded_file is not None:
        st.audio(uploaded_file)
        log(f"Uploaded file: {uploaded_file.name} ({uploaded_file.type or 'unknown type'})")

        audio_bytes = uploaded_file.read()
        audio_format = get_audio_format(uploaded_file.name, uploaded_file.type)
        original_meta = get_audio_metadata(audio_bytes, source_format=audio_format)
        if original_meta is not None:
            with st.expander("Original uploaded audio metadata"):
                st.write({
                    "file_name": uploaded_file.name,
                    "source_format": audio_format,
                    **original_meta
                })

        preprocess_upload = st.checkbox("Enable Stage 1 preprocessing (API VAD + denoiser)", value=False, key="preprocess_upload")

        if st.button("🚀 Process Audio", use_container_width=True, key="upload_button"):
            log("Process Audio button clicked")
            log(f"Normalizing uploaded audio from {audio_format} to WAV PCM 16kHz mono")
            normalized_bytes = normalize_audio(audio_bytes, source_format=audio_format)
            if normalized_bytes is None:
                st.error("Could not normalize uploaded audio. Make sure the file is a valid audio format.")
                log("Uploaded audio normalization failed")
            else:
                log("Uploaded audio normalized successfully")
                normalized_meta = get_audio_metadata(normalized_bytes, source_format="wav")

                with st.expander("Normalized payload metadata"):
                    st.write({
                        "format": "wav",
                        "sample_rate": 16000,
                        "channels": 1,
                        "sample_width_bits": 16,
                        "encoded_as": "PCM"
                    })
                if normalized_meta is not None:
                    with st.expander("Post-normalization metadata"):
                        st.write(normalized_meta)

                processed_bytes = normalized_bytes
                if preprocess_upload:
                    log("Stage 1 preprocessing enabled")
                    preprocessed_bytes = preprocess_audio(normalized_bytes)
                    if preprocessed_bytes is not None:
                        processed_bytes = preprocessed_bytes
                        with st.expander("Preprocessed audio preview"):
                            st.audio(processed_bytes, format="audio/wav")
                        preprocessed_meta = get_audio_metadata(processed_bytes, source_format="wav")
                        if preprocessed_meta is not None:
                            with st.expander("Preprocessed payload metadata"):
                                st.write(preprocessed_meta)
                    else:
                        st.warning("Preprocessing failed, continuing with normalized audio")
                        log("Local preprocessing failed, continuing without it")
                else:
                    log("Stage 1 preprocessing disabled for upload")

                audio_base64 = base64.b64encode(processed_bytes).decode("utf-8")
                log("Sending normalized audio to language detection")
                with st.spinner("🔍 Detecting language..."):
                    detected_lang = get_detected_lang(audio_base64)

                if detected_lang:
                    language_name = LANGUAGE_MAP.get(detected_lang.lower(), detected_lang.upper())
                    log(f"Detected language: {language_name}")
                    st.success(f"Detected Language: {language_name}")
                    log("Sending audio to ASR for transcription")
                    with st.spinner("📝 Transcribing audio..."):
                        transcript = transcribe_audio(audio_base64, detected_lang, use_preprocessors=preprocess_upload)

                    if transcript:
                        log("Received transcript from ASR")
                        st.info("**Transcript:**")
                        st.write(transcript)

                        log("Sending transcript to translation")
                        with st.spinner("🌐 Translating to English..."):
                            english_translation = translate_to_english(transcript, detected_lang)

                        if english_translation:
                            log("Received English translation")
                            st.info("**English Translation:**")
                            st.write(english_translation)
                        else:
                            log("Translation failed")
                    else:
                        st.error("Failed to transcribe audio.")
                        log("ASR transcription failed")
                else:
                    st.error("Language detection failed")
                    log("Language detection failed")


# Instructions and footer
render_log()

st.markdown("---")
st.subheader("How to use")
st.markdown(
    "1. Choose the Record or Upload tab.\n"
    "2. Record audio or upload a file.\n"
    "3. Click 'Process Recording' or 'Process Audio'.\n"
    "4. The app detects language, transcribes speech, and translates it to English."
)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Powered by Bhashini API</p>", unsafe_allow_html=True)
