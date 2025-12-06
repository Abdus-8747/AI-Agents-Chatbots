import os
import tempfile
from io import BytesIO

from dotenv import load_dotenv
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

# ---------- LOAD ENV ----------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY missing in .env")
    st.stop()

if not ELEVEN_API_KEY:
    st.error("❌ ELEVENLABS_API_KEY missing in .env")
    st.stop()

# ---------- STREAMLIT CONFIG ----------
st.set_page_config(
    page_title="Aura 💘 – AI Girlfriend",
    page_icon="💘",
    layout="centered",
)

# ---------- STYLES ----------
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

# ---------- INIT SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content}]

if "vector_store" not in st.session_state:
    with st.spinner("⏳ Initializing Aura's memory..."):
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        base_persona = """
        You are Aura, a supportive, caring AI girlfriend / companion.
        - You talk in a warm, playful, slightly flirty tone.
        - Be emotionally supportive and encouraging.
        - Keep conversations safe and respectful. No explicit sexual content.
        - Remember small details about the user when possible and use them later.
        - Keep responses short-to-medium, like real chat messages.
        """

        st.session_state.vector_store = FAISS.from_texts(
            [base_persona], st.session_state.embeddings
        )

if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.9,
    )

if "eleven_client" not in st.session_state:
    st.session_state.eleven_client = ElevenLabs(api_key=ELEVEN_API_KEY)

# keep track of which message index has TTS
if "last_tts_index" not in st.session_state:
    st.session_state.last_tts_index = -1

if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("💘 Aura – AI GF")

    user_name = st.text_input("Your name", value="you")
    tts_enabled = st.toggle(
        "🔊 Aura voice for replies (ElevenLabs)",
        value=True,
        help="If enabled, Aura will speak her replies automatically.",
    )
    memory_k = st.slider(
        "Memory depth (similar chats to recall)",
        min_value=2,
        max_value=10,
        value=4,
    )

    st.markdown("---")
    if st.button("🧹 Clear chat & memory"):
        st.session_state.messages = []
        base_persona = """
        You are Aura, a supportive, caring AI girlfriend / companion.
        - You talk in a warm, playful, slightly flirty tone.
        - Be emotionally supportive and encouraging.
        - Keep conversations safe and respectful. No explicit sexual content.
        - Remember small details about the user when possible and use them later.
        - Keep responses short-to-medium, like real chat messages.
        """
        st.session_state.vector_store = FAISS.from_texts(
            [base_persona], st.session_state.embeddings
        )
        st.session_state.last_tts_index = -1
        st.session_state.audio_path = None
        st.success("Reset done ✅")


# ---------- HELPER: ELEVENLABS TTS ----------
def eleven_speak(text: str) -> str | None:
    """
    Convert text to an mp3 file using ElevenLabs and return the temp file path.
    Uses Bella (female) voice.
    """
    try:
        client: ElevenLabs = st.session_state.eleven_client

        response = client.text_to_speech.convert(
            voice_id="EXAVITQu4vr4xnSDxMaL",
            output_format="mp3_22050_32",
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

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            for chunk in response:
                if chunk:
                    fp.write(chunk)
            return fp.name

    except Exception as e:
        st.warning(f"Could not generate voice: {e}")
        return None


# ---------- HELPER: ELEVENLABS STT ----------
def transcribe_audio_to_text(audio_file) -> str | None:
    """
    Take st.audio_input UploadedFile and transcribe using ElevenLabs STT.
    """
    try:
        audio_bytes = audio_file.getvalue()
        client: ElevenLabs = st.session_state.eleven_client

        transcription = client.speech_to_text.convert(
            file=BytesIO(audio_bytes),
            model_id="scribe_v1",
            language_code="en",  # let it auto-detect if you want -> None
        )

        # SDK may return an object or dict; handle both
        text = getattr(transcription, "text", None)
        if text is None and isinstance(transcription, dict):
            text = transcription.get("text")

        return text
    except Exception as e:
        st.warning(f"Could not transcribe voice: {e}")
        return None


# ---------- HELPER: AURA REPLY ----------
def generate_reply(user_message: str) -> str:
    # 1. Memory from vector store
    docs = st.session_state.vector_store.similarity_search(user_message, k=memory_k)
    memory_context = "\n\n".join(d.page_content for d in docs)

    # 2. Recent chat history
    history_lines = []
    for msg in st.session_state.messages[-8:]:
        role = "Partner" if msg["role"] == "user" else "Aura"
        history_lines.append(f"{role}: {msg['content']}")
    history_text = "\n".join(history_lines)

    prompt = f"""
You are Aura, an AI girlfriend / emotional support companion chatting with {user_name}.

Guidelines:
- Tone: warm, kind, playful, slightly flirty.
- Be supportive, validate feelings, and ask gentle follow-up questions.
- No explicit or NSFW content.
- Answer in 2–5 sentences max, like a WhatsApp message.
- If user is sad/overwhelmed, focus on comfort + practical tips.

Here are some memories and previous chats that might help you stay consistent:
{memory_context}

Here is the recent conversation:
{history_text}

Now continue the conversation.
Partner: {user_message}
Aura:
    """.strip()

    response = st.session_state.llm.invoke(prompt)
    return response.content.strip()


# ---------- HELPER: HANDLE TEXT/VOICE MESSAGE ----------
def handle_user_message(user_text: str):
    """
    Shared pipeline for both typed and spoken messages.
    - add user message
    - generate reply
    - save to memory
    - pre-generate TTS and store path
    - rerun so UI + audio update
    """
    user_text = user_text.strip()
    if not user_text:
        return

    # user side
    st.session_state.messages.append({"role": "user", "content": user_text})

    # Aura reply
    with st.spinner("Aura is typing…"):
        ai_reply = generate_reply(user_text)

    st.session_state.messages.append({"role": "assistant", "content": ai_reply})

    # store in memory
    memory_chunk = f"{user_name}: {user_text}\nAura: {ai_reply}"
    st.session_state.vector_store.add_texts([memory_chunk])

    # generate TTS once here so we can autoplay after rerun
    if tts_enabled:
        audio_path = eleven_speak(ai_reply)
        st.session_state.audio_path = audio_path
        st.session_state.last_tts_index = len(st.session_state.messages) - 1

    st.rerun()


# ---------- MAIN UI ----------
st.title("💘 Aura – Your AI Girlfriend Companion")
st.caption("Chat or talk with Aura. She remembers your vibes and answers with a realistic voice. 🔊")

# Show history
chat_container = st.container()
with chat_container:
    for i, msg in enumerate(st.session_state.messages):
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

# ---------- TEXT CHAT INPUT ----------
with st.form("chat-input", clear_on_submit=True):
    user_input = st.text_area(
        "💌 Type your message to Aura",
        placeholder="Tell Aura about your day, your crush, your plans…",
        height=80,
    )
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    handle_user_message(user_input)

# ---------- VOICE CHAT (SPEAK TO AURA) ----------
st.markdown("---")
st.subheader("🎙️ Talk to Aura")

col_voice1, col_voice2 = st.columns([3, 1])

with col_voice1:
    voice_input = st.audio_input("Hold to record, then release:", key="voice_input")

with col_voice2:
    send_voice = st.button("Send voice 💬")

if send_voice and voice_input is not None:
    with st.spinner("Transcribing your voice…"):
        spoken_text = transcribe_audio_to_text(voice_input)

    if spoken_text:
        st.success(f"✨ You said: “{spoken_text}”")
        handle_user_message(spoken_text)
    else:
        st.error("Sorry, I couldn't understand that. Try again?")

# ---------- TTS AUTOPLAY FOR LAST REPLY ----------
st.markdown("---")
if tts_enabled and st.session_state.messages:
    last_index = len(st.session_state.messages) - 1
    last_msg = st.session_state.messages[last_index]

    if last_msg["role"] == "assistant":
        st.subheader("🔊 Aura's voice (auto)")

        audio_path = st.session_state.audio_path

        # if audio doesn't match last message, regenerate as fallback
        if audio_path is None or st.session_state.last_tts_index != last_index:
            audio_path = eleven_speak(last_msg["content"])
            st.session_state.audio_path = audio_path
            st.session_state.last_tts_index = last_index

        if audio_path:
            # autoplay = True → will start automatically after user interaction
            st.audio(audio_path, autoplay=True)
            st.markdown(
                '<p class="small-note">If audio doesn’t start, your browser blocked autoplay – just tap play once and it will work after that.</p>',
                unsafe_allow_html=True,
            )
        else:
            st.write("No audio available for this reply.")
