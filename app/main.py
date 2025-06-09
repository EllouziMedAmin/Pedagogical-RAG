from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from core.assistant import PedagogicalAssistant
import whisper
import base64
import uuid
import tempfile
import os
import traceback

print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
print("ELEVENLABS KEY:", os.getenv("ELEVENLABS_API_KEY"))

app = FastAPI()

# Load Whisper model at startup
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded.")

sessions = {}

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Session:
    def __init__(self, name, age, subject):
        try:
            print(f"Creating PedagogicalAssistant for {name}, {age}, {subject}")
            self.assistant = PedagogicalAssistant(
                name, age, subject,
                os.getenv("OPENAI_API_KEY"),
                os.getenv("ELEVENLABS_API_KEY")
            )
            print("Building assistant graph...")
            self.graph = self.assistant.build_graph()
            print("Graph built successfully.")
        except Exception as e:
            print("Error creating session or building graph:")
            traceback.print_exc()
            raise

def transcribe_audio(audio_file, language="fr"):
    try:
        print(f"Transcribing audio with language={language}")
        result = whisper_model.transcribe(audio_file, language=language)
        return result["text"].strip()
    except Exception as e:
        print("Whisper error:", e)
        traceback.print_exc()
        return ""

@app.get("/health")
def health_check():
    return {
        "status": "active",
        "sessions": len(sessions),
        "model": "whisper-base"
    }

@app.post("/session")
def create_session(
    name: str = Form(...),
    age: str = Form(...),
    subject: str = Form(...)
):
    try:
        print(f"Received new session request for {name}, age {age}, subject {subject}")
        session_id = str(uuid.uuid4())
        sessions[session_id] = Session(name, age, subject)
        print(f"Session {session_id} created successfully.")
        return JSONResponse(content={"session_id": session_id})
    except Exception as e:
        print("Error in /session:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to create session")

@app.post("/session/{session_id}/interact")
async def interact(
    session_id: str,
    text: str = Form(None),
    audio: UploadFile = File(None),
    image: UploadFile = File(None)
):
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        image_base64 = None
        user_input = text or ""
        print(f"Interacting with session {session_id}...")

        # Process audio
        if audio:
            print("Processing audio input...")
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                content = await audio.read()
                tmp.write(content)
                tmp.flush()
                language = "en" if session.assistant.is_english_teacher else "fr"
                user_input = transcribe_audio(tmp.name, language)
            print(f"Transcribed audio: {user_input}")

        # Process image
        if image:
            print("Processing image input...")
            content = await image.read()
            image_base64 = base64.b64encode(content).decode("utf-8")

        # Validate input
        if not user_input.strip() and not image_base64:
            return JSONResponse(
                content={"error": "No valid input provided"},
                status_code=400
            )

        print("Invoking LangGraph with input...")
        output_state = session.graph.invoke({
            "user_input": user_input,
            "image_base64": image_base64
        })
        print("LangGraph response:", output_state)

        audio_bytes = session.assistant.synthesize_audio(output_state["response"])
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return JSONResponse(content={
            "text": output_state["response"],
            "audio": audio_b64,
            "format": "audio/mp3"
        })

    except Exception as e:
        print("Error during interaction:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
