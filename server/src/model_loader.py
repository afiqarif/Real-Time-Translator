# model_loader.py
import sys
import os
import torch

def load_models():
    print(">>> 📥 INITIALIZING MODELS...")
    
    # 1. Add Submodules to Path (Crucial for CosyVoice)
    sys.path.append(os.path.abspath("CosyVoice"))
    sys.path.append(os.path.abspath("Matcha-TTS"))
    
    # 2. Import CosyVoice (Only works after sys.path is fixed)
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2
        
        # Load in FP16 for GPU acceleration
        model = CosyVoice2(
            'iic/CosyVoice2-0.5B', 
            load_jit=False, 
            load_trt=False, 
            fp16=True
        )
        print(">>> ✅ COSYVOICE READY")
        return model
    except ImportError as e:
        print(f">>> ❌ FAILED TO LOAD COSYVOICE: {e}")
        return None