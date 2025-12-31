# client/analytics/logger.py
import csv
import time
import os

class BenchmarkLogger:
    """
    Handles logging of latency metrics to a CSV file on the Client.
    """
    def __init__(self, filename="benchmark_results.csv"):
        self.filename = filename
        self.headers = ["Timestamp", "Phrase", "Step_1_Transcription_ms", "Step_2_Server_Wait_ms", "Total_Latency_ms"]
        
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_event(self, phrase, t_start, t_transcribed, t_first_audio):
        stt_duration = (t_transcribed - t_start) * 1000
        network_wait = (t_first_audio - t_transcribed) * 1000
        total_latency = (t_first_audio - t_start) * 1000

        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%H:%M:%S"),
                phrase,
                f"{stt_duration:.2f}",
                f"{network_wait:.2f}",
                f"{total_latency:.2f}"
            ])
        
        print(f"📊 [CLIENT] Total Latency: {total_latency:.0f}ms")