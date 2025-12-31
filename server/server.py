# server.py
import uvicorn
import soundfile as sf
import io
import time
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse

# Import our modules
from src.model_loader import load_models
from src.stream_logic import run_hybrid_stream

app = FastAPI()

# Global Model Variable
cosy_model = load_models()

@app.post("/process_stream")
async def process_stream(
    text: str = Form(...),
    target_lang: str = Form(...),
    prompt_text: str = Form(...),
    audio: UploadFile = File(...)
):
    # Save the reference audio temporarily
    # (CosyVoice needs a file path, unfortunately)
    temp_filename = f"temp_{int(time.time())}.wav"
    audio_bytes = await audio.read()
    data, sr = sf.read(io.BytesIO(audio_bytes))
    sf.write(temp_filename, data, 16000)

    # Return the streaming response using our logic module
    return StreamingResponse(
        run_hybrid_stream(text, target_lang, prompt_text, temp_filename, cosy_model),
        media_type="application/octet-stream"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)