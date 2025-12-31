# gui.py
import customtkinter as ctk
import threading
import os
import pyaudio
from config import LANG_CODES, load_config, save_config
from audio_logic import AudioLogic

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ProApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CosyVoice Translator Pro")
        self.geometry("500x750")
        
        self.config = load_config()
        self.is_reversed = False 
        
        # Initialize Logic (Pass self so logic can log to UI)
        self.logic = AudioLogic(self)
        
        self._setup_ui()
        self.refresh_voice_list()

    def _setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # 1. Settings Frame
        self.frame_settings = ctk.CTkFrame(self)
        self.frame_settings.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.frame_settings, text="Server URL:").grid(row=0, column=0, padx=5)
        self.entry_url = ctk.CTkEntry(self.frame_settings, placeholder_text="https://...", width=300)
        self.entry_url.grid(row=0, column=1, padx=5)
        self.entry_url.insert(0, self.config.get("lightning_url", ""))
        
        ctk.CTkButton(self.frame_settings, text="💾", width=40, command=self.save_settings).grid(row=0, column=2, padx=5)

        # 2. Chat Box
        self.chat_box = ctk.CTkTextbox(self, font=("Segoe UI", 16), state="disabled", wrap="word")
        self.chat_box.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.chat_box.tag_config("user", foreground="#4da6ff", justify="right") 
        self.chat_box.tag_config("ai", foreground="#00e676", justify="left")
        self.chat_box.tag_config("system", foreground="gray", justify="center")
        self.chat_box.tag_config("error", foreground="#ff5252", justify="center")

        # 3. Controls
        self.frame_controls = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_controls.grid(row=2, column=0, padx=10, pady=20, sticky="ew")

        # Language
        ctk.CTkLabel(self.frame_controls, text="Target Language:").pack(side="top", pady=(0, 2))
        self.lang_frame = ctk.CTkFrame(self.frame_controls, fg_color="transparent")
        self.lang_frame.pack(side="top", pady=(0, 10))
        
        self.lang_dropdown = ctk.CTkOptionMenu(self.lang_frame, values=list(LANG_CODES.keys()))
        self.lang_dropdown.set(self.config.get("target_lang", "Japanese"))
        self.lang_dropdown.pack(side="left", padx=5)
        
        self.btn_swap = ctk.CTkButton(self.lang_frame, text="🔁", width=40, fg_color="gray", command=self.toggle_direction)
        self.btn_swap.pack(side="left", padx=5)

        # Voice Profile
        self.frame_voice = ctk.CTkFrame(self.frame_controls, fg_color="transparent")
        self.frame_voice.pack(side="top", fill="x", pady=5)
        ctk.CTkLabel(self.frame_voice, text="Voice Profile:").pack(side="top")
        
        self.voice_dropdown = ctk.CTkOptionMenu(self.frame_voice, values=["Record New..."], command=self.on_voice_change)
        self.voice_dropdown.pack(side="left", padx=5, expand=True)
        
        ctk.CTkButton(self.frame_voice, text="💾", width=40, command=self.on_save_voice).pack(side="right", padx=5)

        # Mic
        ctk.CTkLabel(self.frame_controls, text="Microphone:").pack(side="top", pady=(10, 2))
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
        
        # Status
        self.lbl_status = ctk.CTkLabel(self, text="Status: IDLE", fg_color="gray20", height=30)
        self.lbl_status.grid(row=3, column=0, sticky="ew")

    # --- Event Handlers ---
    def toggle_direction(self):
        self.is_reversed = not self.is_reversed
        if self.is_reversed:
            self.btn_swap.configure(fg_color="orange")
            self.log("system", "🔄 Mode: Foreign -> English")
        else:
            self.btn_swap.configure(fg_color="gray")
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
            self.save_settings()
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
            self.logic.start_listening(mic_idx, url, self.lang_dropdown.get())
        else:
            self.toggle_session()

    def save_settings(self):
        self.config["lightning_url"] = self.entry_url.get().strip()
        self.config["target_lang"] = self.lang_dropdown.get()
        save_config(self.config)

    def refresh_voice_list(self):
        if not os.path.exists("voices"): os.makedirs("voices")
        files = [f for f in os.listdir("voices") if f.endswith(".wav")]
        self.voice_dropdown.configure(values=["Record New..."] + files)

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
        if name and self.logic.save_profile(name):
            self.log("system", f"💾 Saved: {name}")
            self.refresh_voice_list()
            self.voice_dropdown.set(f"{name}.wav")

    # --- Thread-Safe Helpers ---
    def set_status(self, status):
        self.after(0, lambda: self.lbl_status.configure(text=f"Status: {status.upper()}"))

    def log(self, tag, message, newline=True):
        self.after(0, lambda: self._log_impl(tag, message, newline))

    def _log_impl(self, tag, message, newline):
        self.chat_box.configure(state="normal")
        prefix = "\n" if newline else ""
        self.chat_box.insert("end", f"{prefix}{message}", tag)
        if newline: self.chat_box.insert("end", "\n")
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def append_text(self, text):
        self.after(0, lambda: self._log_impl("ai", text, False))