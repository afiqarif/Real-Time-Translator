import time
import struct
import numpy as np
import os
from groq import Groq
from .config import GROQ_API_KEY
from .analytics.logger import ServerLogger

# Initialize global objects
groq_client = Groq(api_key=GROQ_API_KEY)
analytics = ServerLogger() 

def pack_data(type_id: int, data: bytes):
    """
    Helper to pack binary data: [Type(1B)][Length(4B)][Payload]
    """
    length = len(data)
    return struct.pack('>BI', type_id, length) + data

def run_hybrid_stream(text, target_lang, prompt_text, prompt_audio_path, cosy_model):
    """
    The main generator logic. It yields chunks of Text and Audio.
    """
    t_request_start = time.time()
    
    # 1. START TRY BLOCK (Catches errors & ensures cleanup)
    try:
        # --- A. LLM STREAM (Translation) ---
        full_translation = ""
        t_llm_start = time.time()
        
        try:
            stream = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system", 
                        "content": f"Translate English to natural {target_lang}. Return ONLY the translated text. Do not add notes."
                    },
                    {"role": "user", "content": text}
                ],
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_translation += token
                    # Yield Text Token (Type 0)
                    yield pack_data(0, token.encode('utf-8'))
                    
        except Exception as e:
            error_msg = f"LLM Error: {e}"
            print(error_msg)
            yield pack_data(0, error_msg.encode('utf-8'))
            return

        t_llm_end = time.time()

        # --- B. TTS STREAM (Voice Cloning) ---
        t_tts_start = time.time()
        
        try:
            # We use the full translation we just built
            model_output = cosy_model.inference_zero_shot(
                full_translation, 
                prompt_text, 
                prompt_audio_path, 
                stream=True
            )
            
            for chunk in model_output:
                audio_chunk = chunk['tts_speech'].cpu().numpy()
                # Convert Float32 to Int16 PCM
                pcm_chunk = (audio_chunk * 32767).astype(np.int16).tobytes()
                # Yield Audio Chunk (Type 1)
                yield pack_data(1, pcm_chunk)
                
        except Exception as e:
            print(f"TTS Error: {e}")
        
        # --- C. LOG ANALYTICS ---
        t_end = time.time()
        
        # Use the logger class to save data
        analytics.log_transaction(
            text, 
            t_request_start, 
            t_llm_start, 
            t_llm_end, 
            t_tts_start, 
            t_end
        )

    finally:
        # --- D. CLEANUP ---
        # This always runs to delete the temp file
        if os.path.exists(prompt_audio_path):
            try:
                os.remove(prompt_audio_path)
                print(f"🧹 Cleaned up: {prompt_audio_path}")
            except Exception as e:
                print(f"⚠️ Failed to delete temp file: {e}")