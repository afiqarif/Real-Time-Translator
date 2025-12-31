# server/analytics/logger.py
import csv
import time
import os

class ServerLogger:
    """
    Handles logging of processing time metrics on the Server (Cloud).
    """
    def __init__(self, filename="server_metrics.csv"):
        self.filename = filename
        # Check existence to write headers only once
        if not os.path.isfile(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Phrase", "LLM_Time_ms", "TTS_Time_ms", "Total_Time_ms"])

    def log_transaction(self, text, t_start, t_llm_start, t_llm_end, t_tts_start, t_end):
        """
        Calculates durations and saves them to CSV.
        """
        llm_dur = (t_llm_end - t_llm_start) * 1000
        tts_dur = (t_end - t_tts_start) * 1000
        total_dur = (t_end - t_start) * 1000

        # Write to CSV
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%H:%M:%S"),
                text,
                f"{llm_dur:.2f}",
                f"{tts_dur:.2f}",
                f"{total_dur:.2f}"
            ])

        # Print Console Report
        print(f"📊 [SERVER] '{text}'")
        print(f"   ➤ LLM: {llm_dur:.0f}ms | TTS: {tts_dur:.0f}ms | TOTAL: {total_dur:.0f}ms")
        print("------------------------------------------------")