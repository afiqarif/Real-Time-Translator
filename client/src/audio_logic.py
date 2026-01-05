# audio_logic.py
import threading
import queue
import time
import pyaudio
import numpy as np
import sounddevice as sd
import requests
import io
import soundfile as sf
import re
import os
import struct
from funasr import AutoModel
from .analytics.logger import BenchmarkLogger
from .config import LANG_CODES

class AudioLogic:
    """
    Handles all audio processing: Mic input, Local STT, Network, and Playback.
    """
    def __init__(self, gui):
        self.gui = gui
        self.running = False
        self.mic_paused = False
        
        # Voice Profile State
        self.voice_locked = False
        self.reference_audio = None # Numpy Array (for editing)
        self.locked_wav_bytes = None # Cached Bytes (for sending) ⚡
        self.reference_text = ""
        
        self.audio_queue = queue.Queue()
        self.p = pyaudio.PyAudio()
        self.asr_model = None
        
        # Analytics
        self.logger = BenchmarkLogger()
        self.t_silence_detected = 0
        self.t_transcription_done = 0

    def load_models(self):
        try:
            self.gui.log("system", "📥 Loading Model (Hugging Face)...")
            self.asr_model = AutoModel(
                model="FunAudioLLM/SenseVoiceSmall",
                hub="hf", 
                trust_remote_code=True,
                device="cpu", 
                disable_update=True,
                verbose=False
            )
            self.gui.log("system", "✅ Model Loaded Successfully!")
            return True
        except Exception as e:
            self.gui.log("error", f"❌ Model Load Failed: {e}")
            return False

    def _cache_voice_profile(self, audio_data):
        """Encodes the numpy array to WAV bytes ONCE to save CPU later."""
        try:
            buf = io.BytesIO()
            sf.write(buf, audio_data, 16000, format='WAV')
            self.locked_wav_bytes = buf.getvalue() # Store raw bytes
            self.reference_audio = audio_data
            return True
        except Exception as e:
            print(f"Error caching voice: {e}")
            return False

    # Inside audio_logic.py

    def save_profile(self, name):
        # 1. Sanitize the filename (remove weird characters)
        clean_name = "".join([c for c in name if c.isalnum() or c in " _-"])
        if not clean_name: 
            return None, "Invalid filename."

        # 2. Check if we actually have audio
        if not self.voice_locked: 
            return None, "Voice not locked yet."
        if self.reference_audio is None: 
            return None, "No audio data found in memory."

        try:
            # 3. Create folder if missing
            if not os.path.exists("voices"): 
                os.makedirs("voices")
            
            # 4. Save the file
            filename = f"voices/{clean_name}.wav"
            sf.write(filename, self.reference_audio, 16000)
            
            # 5. Return the ABSOLUTE path so you know exactly where it is
            full_path = os.path.abspath(filename)
            return full_path, None

        except Exception as e:
            return None, str(e)

    def load_profile(self, filename):
        try:
            path = f"voices/{filename}"
            if not os.path.exists(path): return False
            data, sr = sf.read(path)
            
            # Use our helper to cache it immediately
            self.voice_locked = True
            self._cache_voice_profile(data)
            self.reference_text = "Loaded from profile" 
            return True
        except: return False

    def play_audio(self, audio_data, sr):
        """Plays a short sound (blocking)."""
        self.mic_paused = True
        self.gui.set_status("speaking")
        try:
            sd.play(audio_data, sr)
            sd.wait()
        except: pass
        finally:
            time.sleep(0.1)
            self.mic_paused = False
            self.gui.set_status("listening")

    def is_hallucination(self, text):
        return len(text) < 2

    def process_phrase(self, audio_bytes, url, target_lang_name, is_reversed):
        import tempfile
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tf_name = tf.name
        
        if is_reversed:
            source_code = LANG_CODES.get(target_lang_name, "auto")
            server_target = "English"
        else:
            source_code = "en"
            server_target = target_lang_name

        try:
            # Save Mic Input to Temp File (Needed for FunASR)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            sf.write(tf_name, audio_int16, 16000)
            tf.close()

            # 1. Transcribe
            self.gui.set_status(f"Transcribing ({source_code})...")
            res = self.asr_model.generate(input=tf_name, cache={}, language=source_code, use_itn=True)
            text = res[0]['text'].strip()
            text = re.sub(r'<\|.*?\|>', '', text).replace('[', '').replace(']', '')
            self.t_transcription_done = time.time()
            
            if self.is_hallucination(text): return

            # 2. Lock Voice Logic (If not locked)
            if not self.voice_locked:
                if len(text) < 4 or text.lower() in ["hmm", "okay", "yeah"]:
                    self.gui.log("error", "❌ Voice not locked: Phrase too short.")
                    return
                
                self.gui.log("system", f"🎤 Locking Voice: '{text}'")
                raw_audio, _ = sf.read(tf_name)
                
                # Trim and Cache
                trimmed = self.trim_silence(raw_audio)
                self._cache_voice_profile(trimmed) # <--- Optimization Here
                self.reference_text = text
                self.voice_locked = True
                
                # Play Beep
                fs = 44100
                tone = (0.5 * np.sin(2*np.pi*np.arange(fs*0.2)*880/fs)).astype(np.float32)
                self.play_audio(tone, fs)
                self.gui.log("system", "✅ READY! Streaming Mode.")
                return

            # 3. Send to Server (Optimized)
            self.gui.log("user", f"({source_code}) {text}") 
            self.gui.set_status("translating")
            self.mic_paused = True 
            
            # ⚡ USE CACHED BYTES (Zero CPU)
            # We wrap the pre-calculated bytes in a IO buffer
            ref_file_obj = io.BytesIO(self.locked_wav_bytes)
            ref_file_obj.name = "ref.wav" # Requests needs a name hint
            
            try:
                response = requests.post(
                    f"{url}/process_stream",
                    data={'text': text, 'target_lang': server_target, 'prompt_text': self.reference_text},
                    files={'audio': ('ref.wav', ref_file_obj, 'audio/wav')},
                    stream=True, 
                    timeout=30
                )
                
                if response.status_code == 200:
                    self._handle_streaming_response(response, text)
                else:
                    self.gui.log("error", f"Server: {response.status_code}")

            except Exception as e:
                self.gui.log("error", f"Stream Error: {e}")

        except Exception as e:
            self.gui.log("error", f"Process Error: {e}")
            
        finally:
            try:
                if os.path.exists(tf_name): os.unlink(tf_name)
            except: pass
            self.mic_paused = False
            self.gui.set_status("listening")

    def _handle_streaming_response(self, response, original_text):
        """Helper to handle the incoming binary stream"""
        raw_stream = response.raw 
        playback_queue = queue.Queue()
        first_audio_received = False
        
        # Background Player
        def playback_worker():
            try:
                with sd.RawOutputStream(samplerate=24000, channels=1, dtype='int16', latency='low', blocksize=1024) as stream:
                    while True:
                        chunk = playback_queue.get()
                        if chunk is None: break 
                        stream.write(chunk)
            except: pass

        player_thread = threading.Thread(target=playback_worker)
        player_thread.start()

        header_buffer = b""
        first_text_token = True
        
        while True:
            header = raw_stream.read(5)
            if not header: break 
            msg_type, msg_len = struct.unpack('>BI', header)
            
            payload = b""
            while len(payload) < msg_len:
                chunk = raw_stream.read(msg_len - len(payload))
                if not chunk: break
                payload += chunk
                
            if msg_type == 0: # Text
                token = payload.decode('utf-8')
                if first_text_token:
                    self.gui.log("ai", f" ", newline=True)
                    first_text_token = False
                self.gui.append_text(token)
                
            elif msg_type == 1: # Audio
                if not first_audio_received:
                    t_arrival = time.time()
                    self.logger.log_event(original_text, self.t_silence_detected, self.t_transcription_done, t_arrival)
                    first_audio_received = True
                playback_queue.put(payload)

        playback_queue.put(None) 
        player_thread.join()

    def start_listening(self, device_idx, url, target_lang):
        self.running = True
        stream = self.p.open(
            format=pyaudio.paInt16, 
            channels=1, 
            rate=16000, 
            input=True, 
            input_device_index=device_idx,
            frames_per_buffer=1024
        )
        self.gui.set_status("listening")
        is_speaking = False
        silence = 0
        phrase = b""
        
        while self.running:
            if self.mic_paused:
                time.sleep(0.05)
                continue
            try:
                data = stream.read(1024, exception_on_overflow=False)
                vol = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
                if vol > 200: 
                    is_speaking = True
                    phrase += data
                    silence = 0
                elif is_speaking:
                    silence += 1
                    phrase += data
                    if silence > 5: 
                        self.t_silence_detected = time.time()
                        current_reversed_state = self.gui.is_reversed
                        threading.Thread(
                            target=self.process_phrase, 
                            args=(phrase, url, target_lang, current_reversed_state)
                        ).start()
                        phrase = b""
                        is_speaking = False
                        silence = 0
            except: pass
        stream.stop_stream()
        stream.close()
    
    def trim_silence(self, audio_data, threshold=0.01):
        """Trims leading/trailing silence from numpy audio array."""
        mask = np.abs(audio_data) > threshold
        if not np.any(mask): return audio_data
        start = np.argmax(mask)
        end = len(mask) - np.argmax(mask[::-1])
        return audio_data[start:end]