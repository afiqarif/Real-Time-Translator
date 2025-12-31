# config.py
import os

# Try to get from environment, or use a default (but don't commit real keys to GitHub!)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "PASTE_YOUR_GROQ_KEY_HERE")

# Model configuration
COSY_MODEL_PATH = 'iic/CosyVoice2-0.5B'