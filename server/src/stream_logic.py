# server/stream_logic.py
import time
import struct
import numpy as np
import os
from groq import Groq
from .config import GROQ_API_KEY
from .analytics.logger import ServerLogger # <--- Import new logger

groq_client = Groq(api_key=GROQ_API_KEY)
# Initialize logger once globally (or inside function if you prefer)
analytics = ServerLogger() 

def pack_data(type_id: int, data: bytes):
    length = len(data)
    return struct.pack('>BI', type_id, length) + data

def run_hybrid_stream(text, target_lang, prompt_text, prompt_audio_path, cosy_model):
    t_request_start = time.time()
    
    try:
        # --- A. LLM STREAM ---
        t_llm_start = time.time()
        # ... (Your LLM Code) ...
        t_llm_end = time.time()

        # --- B. TTS STREAM ---
        t_tts_start = time.time()
        # ... (Your TTS Code) ...
        
        # --- C. LOG ANALYTICS (Clean!) ---
        t_end = time.time()
        
        # Call the new logger class instead of writing CSV manually here
        analytics.log_transaction(
            text, 
            t_request_start, 
            t_llm_start, 
            t_llm_end, 
            t_tts_start, 
            t_end
        )

    finally:
        # Cleanup temp file
        if os.path.exists(prompt_audio_path):
            try:
                os.remove(prompt_audio_path)
            except: pass