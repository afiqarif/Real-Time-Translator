import time  # For analytics/timing
import os
import sys
import io
import struct
import torch
import soundfile as sf
import numpy as np
import uvicorn
import urllib.parse
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from groq import Groq

# --- PATH SETUP ---
# Add submodules to path so we can import CosyVoice and Matcha-TTS
sys.path.append(os.path.abspath("CosyVoice"))
sys.path.append(os.path.abspath("Matcha-TTS"))
from cosyvoice.cli.cosyvoice import CosyVoice2

# --- CONFIGURATION ---
# NOTE: In production, use os.getenv("GROQ_API_KEY") for security
GROQ_API_KEY = "PASTE_YOUR_GROQ_KEY_HERE"

app = FastAPI()

# --- MODEL LOADING ---
print(">>> LOADING MODELS...")
# Load CosyVoice2 (0.5B parameters) in FP16 mode for faster inference on GPU
cosy_model = CosyVoice2('iic/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=True)
groq_client = Groq(api_key=GROQ_API_KEY)
print(">>> ALL SYSTEMS READY.")

def pack_data(type_id: int, data: bytes):
    """
    Helper to pack data into a custom binary protocol for streaming.
    Format: [Type (1 byte)] + [Length (4 bytes)] + [Payload]
    Types: 0 = Text Token, 1 = Audio PCM Chunk
    """
    length = len(data)
    return struct.pack('>BI', type_id, length) + data

@app.post("/process_stream")
async def process_stream(
    text: str = Form(...),
    target_lang: str = Form(...),
    prompt_text: str = Form(...),
    audio: UploadFile = File(...)
):
    """
    Main pipeline endpoint.
    Receives: Transcription text, target language, and reference audio (voice to clone).
    Returns: A continuous stream of mixed Text and Audio packets.
    """
    
    # 1. Start Server-Side Clock (for Analytics)
    t_request_start = time.time()
    
    # 2. Save the Reference Audio (Voice to Clone) to disk
    # CosyVoice requires a file path for the prompt audio
    temp_filename = "temp_ref.wav"
    audio_bytes = await audio.read()
    data, sr = sf.read(io.BytesIO(audio_bytes))
    sf.write(temp_filename, data, 16000)

    def hybrid_generator():
        """
        Generator function that streams response data chunks.
        It runs the LLM and TTS sequentially but streams their outputs immediately.
        """
        
        # --- A. STREAM TRANSLATION (LLM) ---
        full_translation = ""
        t_llm_start = time.time()
        
        try:
            # Call Groq (Llama 3) for fast translation
            stream = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": f"Translate English to natural {target_lang}. Output ONLY the translation."},
                    {"role": "user", "content": text}
                ],
                stream=True
            )
            
            # Stream text tokens back to client as they arrive
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_translation += token
                    # Send Type 0 (Text) packet
                    yield pack_data(0, token.encode('utf-8'))
                    
        except Exception as e:
            err_msg = f"Error: {str(e)}"
            yield pack_data(0, err_msg.encode('utf-8'))
            return

        # Timing: Mark end of translation / start of synthesis
        t_llm_end = time.time()
        t_tts_start = time.time()

        # --- B. STREAM AUDIO (TTS) ---
        # Now feed the full translated text into CosyVoice
        try:
            model_output = cosy_model.inference_zero_shot(
                full_translation, 
                prompt_text, 
                temp_filename, 
                stream=True # Critical: Enable streaming generation
            )
            
            first_chunk = True
            for chunk in model_output:
                # Capture time of first audio chunk for internal latency check
                if first_chunk:
                    t_tts_first_byte = time.time()
                    first_chunk = False
                    
                # Convert PyTorch tensor to raw PCM bytes
                audio_chunk = chunk['tts_speech'].cpu().numpy()
                pcm_chunk = (audio_chunk * 32767).astype(np.int16).tobytes()
                
                # Send Type 1 (Audio) packet
                yield pack_data(1, pcm_chunk)
                
        except Exception as e:
            print(f"Audio Gen Error: {e}")
            
        # --- C. ANALYTICS REPORT ---
        t_end = time.time()
        
        # Calculate server-side durations
        llm_dur = (t_llm_end - t_llm_start) * 1000
        tts_dur = (t_end - t_tts_start) * 1000
        total_dur = (t_end - t_request_start) * 1000
        
        # Print logs to server console (capture these for your report!)
        print(f"📊 [ANALYTICS] '{text}'")
        print(f"   ➤ LLM (Translate): {llm_dur:.0f}ms")
        print(f"   ➤ TTS (Clone Voice): {tts_dur:.0f}ms")
        print(f"   ➤ TOTAL SERVER TIME: {total_dur:.0f}ms")
        print("------------------------------------------------")

    return StreamingResponse(hybrid_generator(), media_type="application/octet-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)