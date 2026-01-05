import uvicorn
import io
import time
import hashlib
import os
import soundfile as sf
import librosa
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse

# Import our modules
from src.model_loader import load_models
from src.stream_logic import run_hybrid_stream

app = FastAPI()

# Global Model Variable
cosy_model = load_models()

# --- ⚡ THE CACHE CONFIG ---
CACHE_DIR = "audio_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

@app.post("/process_stream")
async def process_stream(
    text: str = Form(...),
    target_lang: str = Form(...),
    prompt_text: str = Form(...),
    audio: UploadFile = File(...)
):
    # 1. Read the raw bytes
    audio_bytes = await audio.read()
    
    # 2. Generate Hash (Fingerprint)
    file_hash = hashlib.md5(audio_bytes).hexdigest()
    
    # Define the permanent cache path
    cached_file_path = os.path.join(CACHE_DIR, f"{file_hash}.wav")

    # 3. Check Cache
    if os.path.exists(cached_file_path):
        # HIT: Use the existing file ⚡
        print(f">>> ⚡ CACHE HIT: Reusing audio profile [{file_hash[:8]}...]")
        final_input_path = cached_file_path
    else:
        # MISS: Process and Save to Cache 🐢
        print(f">>> 🐢 CACHE MISS: Processing new audio [{file_hash[:8]}...]")
        
        # Load and Resample to 16000Hz
        # We use librosa or sf.read. 
        # Since 'audio_bytes' is a raw file, we wrap it in BytesIO
        audio_data, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        
        # Save to the cache folder permanently
        sf.write(cached_file_path, audio_data, 16000)
        
        final_input_path = cached_file_path

    # 4. Run Inference
    # Now we pass the STRING path, which CosyVoice likes.
    return StreamingResponse(
        run_hybrid_stream(text, target_lang, prompt_text, final_input_path, cosy_model),
        media_type="application/octet-stream"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)