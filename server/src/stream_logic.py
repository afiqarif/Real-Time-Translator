import time
import struct
import numpy as np
from groq import Groq
from .config import GROQ_API_KEY
from .analytics.logger import ServerLogger

# Initialize global objects
groq_client = Groq(api_key=GROQ_API_KEY)
analytics = ServerLogger() 

def pack_data(type_id: int, data: bytes):
    length = len(data)
    return struct.pack('>BI', type_id, length) + data

# INPUT is now 'prompt_audio_path' (String)
def run_hybrid_stream(text, target_lang, prompt_text, prompt_audio_path, cosy_model):
    
    t_request_start = time.time()
    
    try:
        # --- A. LLM STREAM ---
        full_translation = ""
        t_llm_start = time.time()
        
        try:
            stream = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system", 
                        "content": f"Translate English to natural {target_lang}. Return ONLY the translated text."
                    },
                    {"role": "user", "content": text}
                ],
                stream=True
            )
            
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_translation += content
                    yield pack_data(0, content.encode('utf-8'))
                    
        except Exception as e:
            print(f"LLM Error: {e}")
            full_translation = text 

        t_llm_end = time.time()

        # --- B. TTS STREAM ---
        t_tts_start = time.time()
        
        try:
            # Pass the FILE PATH string. CosyVoice will open it internally.
            model_output = cosy_model.inference_zero_shot(
                full_translation, 
                prompt_text, 
                prompt_audio_path, # <--- String Path
                stream=True
            )
            
            for chunk in model_output:
                audio_chunk = chunk['tts_speech'].cpu().numpy()
                pcm_chunk = (audio_chunk * 32767).astype(np.int16).tobytes()
                yield pack_data(1, pcm_chunk)
                
        except Exception as e:
            print(f"TTS Error: {e}")
        
        # --- C. ANALYTICS ---
        t_end = time.time()
        analytics.log_transaction(text, t_request_start, t_llm_start, t_llm_end, t_tts_start, t_end)

    finally:
        pass