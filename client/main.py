import customtkinter as ctk
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
import json
import urllib.parse
from funasr import AutoModel

# --- 1. LANGUAGE CODES ---
LANG_CODES = {
    "Japanese": "ja",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Korean": "ko",
    "Mandarin Chinese": "zh",
    "Russian": "ru",
    "English": "en"
}

# --- CONFIGURATION ---
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "lightning_url": "",
    "source_lang": "en",
    "target_lang": "Japanese"
}

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class AudioLogic:
    def __init__(self, gui):
        self.gui = gui
        self.running = False
        self.mic_paused = False
        self.voice_locked = False
        self.audio_queue = queue.Queue()
        self.p = pyaudio.PyAudio()
        
        self.reference_audio = None
        self.reference_text = ""
        self.asr_model = None

    def load_models(self):
        try:
            self.gui.log("system", "📥 Downloading/Loading Model (Hugging Face)...")
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

    def save_profile(self, name):
        if not self.voice_locked or self.reference_audio is None: return False
        if not os.path.exists("voices"): os.makedirs("voices")
        filename = f"voices/{name}.wav"
        sf.write(filename, self.reference_audio, 16000)
        return True

    def load_profile(self, filename):
        try:
            path = f"voices/{filename}"
            if not os.path.exists(path): return False
            data, sr = sf.read(path)
            self.reference_audio = data
            self.reference_text = "Loaded from profile" 
            self.voice_locked = True
            return True
        except: return False

    def play_audio(self, audio_data, sr):
        # Used for the "beep" and locking sound only
        self.mic_paused = True
        self.gui.set_status("speaking")
        with self.audio_queue.mutex: self.audio_queue.queue.clear()
        try:
            sd.play(audio_data, sr)
            sd.wait()
        except: pass
        finally:
            time.sleep(0.1)
            self.mic_paused = False
            self.gui.set_status("listening")

    def clean_audio(self, audio_data, rate=16000):
        return audio_data

    def is_hallucination(self, text):
        if len(text) < 2: return True
        return False

    def process_phrase(self, audio_bytes, url, target_lang_name, is_reversed):
        import tempfile
        import struct
        
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tf_name = tf.name
        
        if is_reversed:
            source_code = LANG_CODES.get(target_lang_name, "auto")
            server_target = "English"
        else:
            source_code = "en"
            server_target = target_lang_name

        try:
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            sf.write(tf_name, audio_int16, 16000)
            tf.close()

            # 1. Transcribe (Local)
            self.gui.set_status(f"Transcribing ({source_code})...")
            res = self.asr_model.generate(input=tf_name, cache={}, language=source_code, use_itn=True)
            text = res[0]['text'].strip()
            text = re.sub(r'<\|.*?\|>', '', text).replace('[', '').replace(']', '')

            if self.is_hallucination(text): return

            # --- LOCK VOICE ---
            if not self.voice_locked:
                self.gui.log("system", f"🎤 Locking Voice: '{text}'")
                self.reference_audio, _ = sf.read(tf_name)
                self.reference_text = text
                self.voice_locked = True
                fs = 44100
                tone = (0.5 * np.sin(2*np.pi*np.arange(fs*0.2)*880/fs)).astype(np.float32)
                self.play_audio(tone, fs)
                self.gui.log("system", "✅ READY! Streaming Mode.")
                return

            # --- SEND TO SERVER ---
            self.gui.log("user", f"({source_code}) {text}") 
            self.gui.set_status("translating")
            self.mic_paused = True 
            
            ref_io = io.BytesIO()
            sf.write(ref_io, self.reference_audio, 16000, format='WAV')
            ref_io.seek(0)
            
            try:
                response = requests.post(
                    f"{url}/process_stream",
                    data={'text': text, 'target_lang': server_target, 'prompt_text': self.reference_text},
                    files={'audio': ('ref.wav', ref_io, 'audio/wav')},
                    stream=True, 
                    timeout=30
                )
                
                if response.status_code == 200:
                    raw_stream = response.raw 
                    
                    # --- NEW: THREADED PLAYER ---
                    # We create a separate queue for playback so the network loop never waits
                    playback_queue = queue.Queue()
                    
                    def playback_worker():
                        # This runs in background: Pulls audio and plays it
                        try:
                            # ADDED: latency='low' and blocksize=1024 for faster response
                            with sd.RawOutputStream(
                                samplerate=24000, 
                                channels=1, 
                                dtype='int16',
                                latency='low',
                                blocksize=1024 
                            ) as stream:
                                while True:
                                    chunk = playback_queue.get()
                                    if chunk is None: break 
                                    stream.write(chunk)
                        except Exception as e:
                            print(f"Player Error: {e}")

                    player_thread = threading.Thread(target=playback_worker)
                    player_thread.start()
                    # ---------------------------

                    header_buffer = b""
                    first_text_token = True
                    
                    while True:
                        # 1. READ HEADER
                        header = raw_stream.read(5)
                        if not header: break 
                        msg_type, msg_len = struct.unpack('>BI', header)
                        
                        # 2. READ PAYLOAD
                        payload = b""
                        while len(payload) < msg_len:
                            chunk = raw_stream.read(msg_len - len(payload))
                            if not chunk: break
                            payload += chunk
                            
                        # 3. HANDLE PACKET (NON-BLOCKING NOW!)
                        if msg_type == 0: # TEXT
                            token = payload.decode('utf-8')
                            if first_text_token:
                                self.gui.log("ai", f"({server_target}) ", newline=True)
                                first_text_token = False
                            self.gui.append_text(token)
                            
                        elif msg_type == 1: # AUDIO
                            # Just throw it in the queue and keep moving! 
                            # Don't wait for it to play.
                            playback_queue.put(payload)

                    # Signal player to stop when stream ends
                    playback_queue.put(None) 
                    player_thread.join() # Wait for audio to finish before unpausing mic
                    
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
            with self.audio_queue.mutex:
                self.audio_queue.queue.clear()

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

class ProApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CosyVoice Translator Pro")
        self.geometry("500x750")
        
        # --- INIT STATE ---
        self.is_reversed = False 
        self.logic = AudioLogic(self)
        self.load_config()
        
        # --- UI LAYOUT ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # A. Settings
        self.frame_settings = ctk.CTkFrame(self)
        self.frame_settings.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.lbl_url = ctk.CTkLabel(self.frame_settings, text="Server URL:")
        self.lbl_url.grid(row=0, column=0, padx=5, pady=5)
        self.entry_url = ctk.CTkEntry(self.frame_settings, placeholder_text="https://...", width=300)
        self.entry_url.grid(row=0, column=1, padx=5, pady=5)
        self.entry_url.insert(0, self.config.get("lightning_url", ""))
        self.btn_save = ctk.CTkButton(self.frame_settings, text="💾", width=40, command=self.save_config)
        self.btn_save.grid(row=0, column=2, padx=5)

        # B. Chat Box
        self.chat_box = ctk.CTkTextbox(self, font=("Segoe UI", 16), state="disabled", wrap="word")
        self.chat_box.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.chat_box.tag_config("user", foreground="#4da6ff", justify="right") 
        self.chat_box.tag_config("ai", foreground="#00e676", justify="left")
        self.chat_box.tag_config("system", foreground="gray", justify="center")
        self.chat_box.tag_config("error", foreground="#ff5252", justify="center")

        # C. Controls Frame
        self.frame_controls = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_controls.grid(row=2, column=0, padx=10, pady=20, sticky="ew")

        # --- LANGUAGE SELECTOR & SWAP ---
        self.lbl_lang = ctk.CTkLabel(self.frame_controls, text="Target Language:")
        self.lbl_lang.pack(side="top", pady=(0, 2))
        
        self.lang_frame = ctk.CTkFrame(self.frame_controls, fg_color="transparent")
        self.lang_frame.pack(side="top", pady=(0, 10))

        self.languages = list(LANG_CODES.keys())
        self.lang_dropdown = ctk.CTkOptionMenu(self.lang_frame, values=self.languages)
        self.lang_dropdown.set("Japanese")
        self.lang_dropdown.pack(side="left", padx=5)

        self.btn_swap = ctk.CTkButton(
            self.lang_frame, 
            text="🔁", 
            width=40, 
            fg_color="gray", 
            command=self.toggle_direction
        )
        self.btn_swap.pack(side="left", padx=5)
        # --------------------------------

        # Voice Profile
        self.frame_voice = ctk.CTkFrame(self.frame_controls, fg_color="transparent")
        self.frame_voice.pack(side="top", fill="x", pady=5)
        self.lbl_voice = ctk.CTkLabel(self.frame_voice, text="Voice Profile:")
        self.lbl_voice.pack(side="top")
        self.voice_dropdown = ctk.CTkOptionMenu(self.frame_voice, values=["Record New..."], command=self.on_voice_change)
        self.voice_dropdown.pack(side="left", padx=5, expand=True)
        self.btn_save_voice = ctk.CTkButton(self.frame_voice, text="💾", width=40, command=self.on_save_voice)
        self.btn_save_voice.pack(side="right", padx=5)

        # Microphone
        self.lbl_mic = ctk.CTkLabel(self.frame_controls, text="Microphone:")
        self.lbl_mic.pack(side="top", pady=(10, 2))
        self.mics = self.get_microphones()
        self.mic_dropdown = ctk.CTkOptionMenu(self.frame_controls, values=list(self.mics.keys()))
        self.mic_dropdown.pack(side="top", pady=5)
        
        # Start Button
        self.btn_toggle = ctk.CTkButton(
            self.frame_controls, 
            text="START LISTENING", 
            font=("Arial", 18, "bold"),
            height=60, 
            fg_color="green",
            command=self.toggle_session
        )
        self.btn_toggle.pack(fill="x", padx=20, pady=10)
        
        # D. Status Bar
        self.lbl_status = ctk.CTkLabel(self, text="Status: IDLE", fg_color="gray20", height=30)
        self.lbl_status.grid(row=3, column=0, sticky="ew")

        self.refresh_voice_list()

    # --- UI EVENT HANDLERS ---
    def toggle_direction(self):
        self.is_reversed = not self.is_reversed
        if self.is_reversed:
            self.btn_swap.configure(fg_color="orange")
            self.lbl_lang.configure(text="Listening to Foreign... (Target: English)")
            self.log("system", "🔄 Mode: Foreign -> English")
        else:
            self.btn_swap.configure(fg_color="gray")
            self.lbl_lang.configure(text="Target Language:")
            self.log("system", "🔄 Mode: English -> Foreign")

    def get_microphones(self):
        p = pyaudio.PyAudio()
        mics = {}
        for i in range(p.get_device_count()):
            try:
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0 and info['hostApi'] == 0:
                    name = info['name']
                    if len(name) > 30: name = name[:27] + "..."
                    mics[name] = i
            except: pass
        return mics

    def toggle_session(self):
        if not self.logic.running:
            url = self.entry_url.get().strip()
            if not url.startswith("http"):
                self.log("error", "Invalid URL!")
                return
            self.save_config()
            self.btn_toggle.configure(text="STOP", fg_color="#d32f2f")
            self.entry_url.configure(state="disabled")
            mic_idx = self.mics.get(self.mic_dropdown.get(), 0)
            threading.Thread(target=self.run_backend, args=(url, mic_idx)).start()
        else:
            self.logic.running = False
            self.btn_toggle.configure(text="START LISTENING", fg_color="green")
            self.entry_url.configure(state="normal")
            self.set_status("IDLE")

    def run_backend(self, url, mic_idx):
        if self.logic.load_models():
            selected_lang = self.lang_dropdown.get()
            # CORRECT CALL: No is_reversed argument passed here!
            self.logic.start_listening(mic_idx, url, selected_lang)
        else:
            self.toggle_session()

    def set_status(self, status):
        if hasattr(self, 'lbl_status'):
            self.after(0, lambda: self._update_status_impl(status))

    def _update_status_impl(self, status):
        self.lbl_status.configure(text=f"Status: {status.upper()}")

    def log(self, tag, message, newline=True): # Added newline arg
        self.after(0, lambda: self._log_impl(tag, message, newline))

    def _log_impl(self, tag, message, newline):
        self.chat_box.configure(state="normal")
        prefix = "\n" if newline else ""
        self.chat_box.insert("end", f"{prefix}{message}", tag)
        if newline: self.chat_box.insert("end", "\n") # Force newline after
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")
    
    def append_text(self, text):
        self.after(0, lambda: self._append_text_impl(text))

    def _append_text_impl(self, text):
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", text, "ai") # Insert at end
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def load_config(self):
        try:
            with open(CONFIG_FILE, "r") as f: self.config = json.load(f)
        except: self.config = DEFAULT_CONFIG

    def save_config(self):
        self.config["lightning_url"] = self.entry_url.get().strip()
        with open(CONFIG_FILE, "w") as f: json.dump(self.config, f)

    def refresh_voice_list(self):
        if not os.path.exists("voices"): os.makedirs("voices")
        files = [f for f in os.listdir("voices") if f.endswith(".wav")]
        menu_items = ["Record New..."] + files
        self.voice_dropdown.configure(values=menu_items)

    def on_voice_change(self, selection):
        if selection == "Record New...":
            self.logic.voice_locked = False
            self.logic.reference_audio = None
            self.log("system", "🔄 Voice Reset. Please speak to lock.")
        else:
            if self.logic.load_profile(selection):
                self.log("system", f"📂 Loaded Voice: {selection}")
            else:
                self.log("error", "Failed to load voice file.")

    def on_save_voice(self):
        if not self.logic.voice_locked:
            self.log("error", "❌ No voice locked yet!")
            return
        dialog = ctk.CTkInputDialog(text="Name this voice profile:", title="Save Voice")
        name = dialog.get_input()
        if name:
            if self.logic.save_profile(name):
                self.log("system", f"💾 Saved: {name}")
                self.refresh_voice_list()
                self.voice_dropdown.set(f"{name}.wav")

if __name__ == "__main__":
    app = ProApp()
    app.mainloop()