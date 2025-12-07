import os
import sys
import time

from dotenv import load_dotenv
import sounddevice as sd
from scipy.io.wavfile import write as write_wav

from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

from langchain_groq import ChatGroq

import numpy as np
import streamlit as st

# ---------- CONFIG ----------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not GROQ_API_KEY or not ELEVEN_API_KEY:
    st.error("Missing GROQ_API_KEY or ELEVENLABS_API_KEY in .env")
    st.stop()

client = ElevenLabs(api_key=ELEVEN_API_KEY)
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.8,
)

USER_NAME = "Abdus"  # change if you want
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # Bella
SAMPLE_RATE = 16000
LISTEN_SECONDS = 5  # how long to listen each turn

# ---------- STREAMLIT CONFIG ----------
st.set_page_config(page_title="Aura Voice 💘", page_icon="💘", layout="centered")

st.markdown(
    """
    <style>
    .chat-bubble-user {
        background: #1f2933;
        color: white;
        padding: 0.6rem 0.8rem;
        border-radius: 0.75rem;
        margin-bottom: 0.35rem;
        max-width: 85%;
        border: 1px solid #4b5563;
    }
    .chat-bubble-ai {
        background: linear-gradient(135deg, #f472b6, #fb7185);
        color: #111827;
        padding: 0.6rem 0.8rem;
        border-radius: 0.75rem;
        margin-bottom: 0.35rem;
        max-width: 85%;
        border: 1px solid #f9a8d4;
    }
    .chat-row {
        display: flex;
        margin: 0.25rem 0;
    }
    .chat-row.user {
        justify-content: flex-end;
    }
    .chat-row.ai {
        justify-content: flex-start;
    }
    .small-note {
        font-size: 0.8rem;
        color: #6b7280;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- SESSION STATE ----------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []  # list[dict]: {role, content}

if "last_reply_audio" not in st.session_state:
    st.session_state.last_reply_audio = None


# ---------- SIMPLE MEMORY ----------
def build_prompt(user_text: str) -> str:
    history_lines = []
    for msg in st.session_state.conversation_history[-8:]:
        role = "Partner" if msg["role"] == "user" else "Aura"
        history_lines.append(f"{role}: {msg['content']}")

    history_text = "\n".join(history_lines)

    persona = f"""
You are Aura, an AI girlfriend / emotional support companion chatting with {USER_NAME}.

Guidelines:
- Tone: warm, kind, playful, slightly flirty (but SFW).
- Be supportive, validate feelings, and ask gentle follow-up questions.
- No explicit or NSFW content.
- Answer in 1–4 sentences, like a short WhatsApp voice note.
- If the user is sad/overwhelmed, focus on comfort + practical tips.

Recent conversation:
{history_text}

Now continue the conversation.
Partner: {user_text}
Aura:
    """.strip()

    return persona


# ---------- AUDIO RECORD (LOCAL MIC) ----------
def record_audio(filename: str, duration: float = LISTEN_SECONDS):
    """Record from system microphone using sounddevice."""
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    write_wav(filename, SAMPLE_RATE, audio)


# ---------- STT ----------
def speech_to_text(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            result = client.speech_to_text.convert(
                file=f,
                model_id="scribe_v1",
                language_code="en",
            )
        text = getattr(result, "text", None)
        if text is None and isinstance(result, dict):
            text = result.get("text")
        return text
    except Exception as e:
        st.warning(f"STT error: {e}")
        return None


# ---------- TTS ----------
def text_to_speech(text: str):
    try:
        response = client.text_to_speech.convert(
            voice_id=VOICE_ID,
            output_format="pcm_16000",  # raw PCM for direct playback
            text=text,
            model_id="eleven_turbo_v2_5",
            voice_settings=VoiceSettings(
                stability=0.3,
                similarity_boost=0.9,
                style=0.3,
                use_speaker_boost=True,
                speed=1.0,
            ),
        )

        audio_bytes = b"".join(chunk for chunk in response if chunk)
        return audio_bytes
    except Exception as e:
        st.warning(f"TTS error: {e}")
        return None


def pcm_to_wav_bytes(audio_bytes: bytes) -> bytes:
    """
    Convert raw 16bit PCM mono 16k audio to WAV bytes
    so Streamlit can play it via st.audio.
    """
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_bytes)
    return buffer.getvalue()


# ---------- AURA PIPELINE ----------
def aura_turn_from_voice():
    # 1) record voice
    tmp_wav = "user_input.wav"
    with st.spinner(f"🎙️ Listening for {LISTEN_SECONDS} seconds..."):
        record_audio(tmp_wav, LISTEN_SECONDS)

    # 2) STT
    with st.spinner("Transcribing your voice..."):
        user_text = speech_to_text(tmp_wav)

    if not user_text:
        st.error("I couldn't hear anything meaningful. Try again?")
        return

    st.info(f"🗣️ You said: {user_text}")

    # 3) LLM
    prompt = build_prompt(user_text)
    with st.spinner("Aura is thinking..."):
        response = llm.invoke(prompt)
    aura_text = response.content.strip()

    # save conversation
    st.session_state.conversation_history.append({"role": "user", "content": user_text})
    st.session_state.conversation_history.append({"role": "assistant", "content": aura_text})

    st.success(f"💬 Aura: {aura_text}")

    # 4) TTS
    with st.spinner("Generating Aura's voice..."):
        audio_bytes = text_to_speech(aura_text)

    if audio_bytes:
        wav_bytes = pcm_to_wav_bytes(audio_bytes)
        st.session_state.last_reply_audio = wav_bytes
    else:
        st.session_state.last_reply_audio = None


# ---------- UI ----------
st.title("💘 Aura – Voice Girlfriend")
st.caption("Talk, pause, and let Aura reply back with her own voice.")

# Chat history display
chat_container = st.container()
with chat_container:
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="chat-row user">
                    <div class="chat-bubble-user">{msg['content']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="chat-row ai">
                    <div class="chat-bubble-ai">{msg['content']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    st.write(f"🎙️ Press the button and speak for {LISTEN_SECONDS} seconds.")
with col2:
    if st.button("Talk to Aura 💬"):
        aura_turn_from_voice()

st.markdown("---")

# Auto-play last reply if available
if st.session_state.last_reply_audio:
    st.subheader("🔊 Aura's latest reply")
    st.audio(st.session_state.last_reply_audio, format="audio/wav", autoplay=True)
    st.markdown(
        '<p class="small-note">If autoplay doesn\'t start, your browser blocked it. Just hit play once.</p>',
        unsafe_allow_html=True,
    )

# Clear conversation
if st.button("🧹 Clear conversation"):
    st.session_state.conversation_history = []
    st.session_state.last_reply_audio = None
    st.experimental_rerun()
