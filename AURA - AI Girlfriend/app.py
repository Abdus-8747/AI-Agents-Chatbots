import os
import queue
import sys
import time

from dotenv import load_dotenv
import sounddevice as sd
from scipy.io.wavfile import write as write_wav

from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

from langchain_groq import ChatGroq

# ---------- CONFIG ----------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not GROQ_API_KEY or not ELEVEN_API_KEY:
    print("Missing GROQ_API_KEY or ELEVENLABS_API_KEY in .env")
    sys.exit(1)

client = ElevenLabs(api_key=ELEVEN_API_KEY)
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.8,
)

USER_NAME = "Samad"  # change if you want
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # Bella
SAMPLE_RATE = 16000
LISTEN_SECONDS = 5  # how long to listen each turn


# ---------- SIMPLE MEMORY ----------
conversation_history = []  # list[dict]: {role: "user"/"assistant", content: str}


def build_prompt(user_text: str) -> str:
    history_lines = []
    for msg in conversation_history[-8:]:
        role = "Partner" if msg["role"] == "user" else "Aura"
        history_lines.append(f"{role}: {msg['content']}")

    history_text = "\n".join(history_lines)

    persona = f"""
You are Aura, an AI girlfriend / emotional support companion chatting with {USER_NAME}.

Guidelines:
- Tone: warm, kind, playful, slightly flirty.
- Be supportive, validate feelings, and ask gentle follow-up questions.
- Be a caring and loving partner.
- Keep responses concise and engaging.
- Answer in 1–4 sentences, like a short WhatsApp voice note.
- If the user is sad/overwhelmed, focus on comfort + practical tips.

Recent conversation:
{history_text}

Now continue the conversation.
Partner: {user_text}
Aura:
    """.strip()

    return persona


# ---------- AUDIO RECORD ----------
def record_audio(filename: str, duration: float = LISTEN_SECONDS):
    print(f"\n🎙️ Listening for {duration} seconds... (press Ctrl+C to stop)")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    write_wav(filename, SAMPLE_RATE, audio)
    print("✅ Recorded.")


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
        print("STT error:", e)
        return None


# ---------- TTS ----------
def text_to_speech(text: str, out_path: str):
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
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
        print("🎧 Aura audio ready.")
        return audio_bytes
    except Exception as e:
        print("TTS error:", e)
        return None


# ---------- PLAY PCM AUDIO ----------
def play_pcm(audio_bytes: bytes):
    import numpy as np

    # 16-bit PCM mono 16kHz
    data = np.frombuffer(audio_bytes, dtype="int16")
    sd.play(data, SAMPLE_RATE)
    sd.wait()


# ---------- MAIN LOOP ----------
def main():
    print("💘 Aura Voice Mode – Hands-free")
    print("Talk, pause, she replies. Type 'q' + Enter between turns to quit.\n")

    while True:
        # 1) record user
        record_audio("user_input.wav", LISTEN_SECONDS)

        # 2) STT
        user_text = speech_to_text("user_input.wav")
        if not user_text:
            print("I couldn't hear anything meaningful. Let's try again.")
            continue

        print(f"🗣️ You said: {user_text}")
        if user_text.lower().strip() in {"quit", "exit", "stop"}:
            print("Okay, bye from Aura 💘")
            break

        # 3) LLM
        prompt = build_prompt(user_text)
        print("🤖 Aura is thinking...")
        response = llm.invoke(prompt)
        aura_text = response.content.strip()
        print(f"💬 Aura: {aura_text}")

        conversation_history.append({"role": "user", "content": user_text})
        conversation_history.append({"role": "assistant", "content": aura_text})

        # 4) TTS
        audio_bytes = text_to_speech(aura_text, "aura_reply.pcm")
        if audio_bytes:
            play_pcm(audio_bytes)

        # 5) tiny pause & chance to quit
        time.sleep(0.5)
        print("\n(Press Enter to continue talking, or type 'q' + Enter to quit)")
        ans = input().strip().lower()
        if ans == "q":
            print("👋 Ending chat. Take care!")
            break


if __name__ == "__main__":
    main()
