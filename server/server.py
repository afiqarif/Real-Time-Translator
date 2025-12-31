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

sys.path.append(os.path.abspath("CosyVoice"))
sys.path.append(os.path.abspath("Matcha-TTS"))
from cosyvoice.cli.cosyvoice import CosyVoice2

# --- CONFIGURATION ---
GROQ_API_KEY = "PASTE_YOUR_GROQ_KEY_HERE"

app = FastAPI()

print(">>> LOADING MODELS...")
cosy_model = CosyVoice2('iic/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=True)
groq_client = Groq(api_key=GROQ_API_KEY)
print(">>> ALL SYSTEMS READY.")

def pack_data(type_id: int, data: bytes):
    # Format: [Type (1B)] + [Length (4B)] + [Data]
    length = len(data)
    return struct.pack('>BI', type_id, length) + data

@app.post("/process_stream")
async def process_stream(
    text: str = Form(...),
    target_lang: str = Form(...),
    prompt_text: str = Form(...),
    audio: UploadFile = File(...)
):
    # 1. SAVE AUDIO REF
    temp_filename = "temp_ref.wav"
    audio_bytes = await audio.read()
    data, sr = sf.read(io.BytesIO(audio_bytes))
    sf.write(temp_filename, data, 16000)

    def hybrid_generator():
        # A. STREAM TRANSLATION (Text Packets)
        full_translation = ""
        try:
            stream = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": f"Translate English to natural {target_lang}. Output ONLY the translation."},
                    {"role": "user", "content": text}
                ],
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_translation += token
                    # TYPE 0 = TEXT
                    yield pack_data(0, token.encode('utf-8'))
                    
        except Exception as e:
            err_msg = f"Error: {str(e)}"
            yield pack_data(0, err_msg.encode('utf-8'))
            return

        # B. STREAM AUDIO (Audio Packets)
        # We need the full text now to generate audio
        try:
            model_output = cosy_model.inference_zero_shot(
                full_translation, 
                prompt_text, 
                temp_filename, 
                stream=True
            )
            
            for chunk in model_output:
                audio_chunk = chunk['tts_speech'].cpu().numpy()
                pcm_chunk = (audio_chunk * 32767).astype(np.int16).tobytes()
                # TYPE 1 = AUDIO
                yield pack_data(1, pcm_chunk)
                
        except Exception as e:
            print(f"Audio Gen Error: {e}")

    return StreamingResponse(hybrid_generator(), media_type="application/octet-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
