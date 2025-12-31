import csv
import time
import os

class BenchmarkLogger:
    """
    Handles logging of performance metrics to a CSV file.
    Used to generate latency reports for the course project.
    """
    def __init__(self, filename="benchmark_results.csv"):
        self.filename = filename
        # Define columns: Timestamp, Input Phrase, and the breakdown of latency steps
        self.headers = ["Timestamp", "Phrase", "Step_1_Transcription_ms", "Step_2_Server_Wait_ms", "Total_Latency_ms"]
        
        # Initialize the CSV file with headers if it doesn't exist yet
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_event(self, phrase, t_start, t_transcribed, t_first_audio):
        """
        Calculates and logs the latency breakdown for a single translation event.
        
        Args:
            phrase (str): The text spoken by the user.
            t_start (float): Timestamp when the user finished speaking (Silence Detected).
            t_transcribed (float): Timestamp when local STT (FunASR) finished.
            t_first_audio (float): Timestamp when the first TTS audio packet arrived from server.
        """
        # 1. Local Processing Time (STT)
        stt_duration = (t_transcribed - t_start) * 1000
        
        # 2. Network + Server Processing Time (The "Black Box" wait)
        network_wait = (t_first_audio - t_transcribed) * 1000
        
        # 3. Total Time-to-First-Audio (The user experience metric)
        total_latency = (t_first_audio - t_start) * 1000

        # Append data to the CSV
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%H:%M:%S"),
                phrase,
                f"{stt_duration:.2f}",
                f"{network_wait:.2f}",
                f"{total_latency:.2f}"
            ])
        
        # Print summary to console for immediate feedback
        print(f"📊 LOGGED: Total Latency = {total_latency:.0f}ms")