import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- CONFIGURATION ---
# We use a helper to find the file whether you run this from 'client/' or 'client/scripts/'
def find_csv(filename):
    # Check current folder
    if os.path.exists(filename): 
        return filename
    # Check one folder up (for when running from scripts/)
    if os.path.exists(f"../{filename}"): 
        return f"../{filename}"
    return None

CLIENT_CSV_PATH = find_csv("benchmark_results.csv")
SERVER_CSV_PATH = find_csv("server_metrics.csv")
OUTPUT_FOLDER = "report_charts"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def clean_phrase(phrase):
    """Truncates long phrases for chart labels."""
    return phrase[:15] + "..." if len(str(phrase)) > 15 else str(phrase)

# --- LOAD DATA ---
print(">>> 📂 Loading Data...")

# 1. LOAD CLIENT DATA
df_client = pd.DataFrame()
if CLIENT_CSV_PATH:
    try:
        df_client = pd.read_csv(CLIENT_CSV_PATH)
        # --- THE FIX: Remove hidden spaces from column names ---
        df_client.columns = df_client.columns.str.strip() 
        print(f"✅ Loaded Client Data: {len(df_client)} samples")
    except Exception as e:
        print(f"❌ Error reading Client CSV: {e}")
else:
    print("⚠️ Client CSV not found.")

# 2. LOAD SERVER DATA
df_server = pd.DataFrame()
if SERVER_CSV_PATH:
    try:
        df_server = pd.read_csv(SERVER_CSV_PATH)
        # --- THE FIX: Remove hidden spaces here too ---
        df_server.columns = df_server.columns.str.strip()
        print(f"✅ Loaded Server Data: {len(df_server)} samples")
    except Exception as e:
        print(f"❌ Error reading Server CSV: {e}")
else:
    print("⚠️ Server CSV not found.")

# ==========================================
# CHART 1: Client Latency Waterfall
# ==========================================
if not df_client.empty:
    try:
        plt.figure(figsize=(10, 6))
        
        # Verify columns exist before plotting to avoid crash
        required = ['Step_1_Transcription_ms', 'Step_2_Server_Wait_ms']
        if all(col in df_client.columns for col in required):
            x = range(len(df_client))
            y1 = df_client['Step_1_Transcription_ms']
            y2 = df_client['Step_2_Server_Wait_ms']
            labels = [clean_phrase(p) for p in df_client['Phrase']]

            plt.bar(x, y1, label='Local STT (Edge)', color='#4da6ff')
            plt.bar(x, y2, bottom=y1, label='Cloud Processing + Network', color='#ff9933')

            plt.xlabel('Test Phrase')
            plt.ylabel('Latency (ms)')
            plt.title('Fig 1: End-to-End System Latency (Client View)')
            plt.xticks(x, labels, rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_FOLDER}/chart_1_client_latency.png")
            print("generated Chart 1...")
        else:
            print(f"⚠️ Missing columns for Chart 1. Found: {list(df_client.columns)}")
    except Exception as e:
        print(f"❌ Error generating Chart 1: {e}")

# ==========================================
# CHART 2: Server Processing Breakdown
# ==========================================
if not df_server.empty:
    try:
        plt.figure(figsize=(8, 8))
        
        avg_llm = df_server['LLM_Time_ms'].mean()
        avg_tts = df_server['TTS_Time_ms'].mean()
        avg_total = df_server['Total_Time_ms'].mean()
        avg_overhead = max(0, avg_total - (avg_llm + avg_tts))

        sizes = [avg_llm, avg_tts, avg_overhead]
        labels = [
            f'Translation\n{avg_llm:.0f} ms', 
            f'Voice Cloning\n{avg_tts:.0f} ms', 
            f'System I/O\n{avg_overhead:.0f} ms'
        ]
        colors = ['#ffcc99', '#ff6666', '#99ff99']

        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.title(f'Fig 2: Server Processing Distribution\n(Avg Total: {avg_total:.0f} ms)')
        
        plt.savefig(f"{OUTPUT_FOLDER}/chart_2_server_breakdown.png")
        print("generated Chart 2...")
    except Exception as e:
        print(f"❌ Error generating Chart 2: {e}")

# ==========================================
# CHART 3: Network vs. Compute
# ==========================================
if not df_client.empty and not df_server.empty:
    try:
        plt.figure(figsize=(6, 6))
        
        avg_client_total = df_client['Total_Latency_ms'].mean()
        avg_server_total = df_server['Total_Time_ms'].mean()
        avg_stt = df_client['Step_1_Transcription_ms'].mean()
        
        avg_network = max(0, avg_client_total - avg_server_total - avg_stt)

        components = [avg_stt, avg_server_total, avg_network]
        comp_labels = ['Local STT', 'Cloud Compute', 'Network Transfer']
        colors = ['#4da6ff', '#ff6666', '#99cc00']
        
        plt.bar(comp_labels, components, color=colors)
        plt.ylabel('Time (ms)')
        plt.title('Fig 3: Latency Composition')
        
        for i, v in enumerate(components):
            plt.text(i, v + 50, f"{v:.0f}ms", ha='center', fontweight='bold')

        plt.savefig(f"{OUTPUT_FOLDER}/chart_3_network_impact.png")
        print("generated Chart 3...")
    except Exception as e:
        print(f"❌ Error generating Chart 3: {e}")

# ==========================================
# CHART 4: Scalability
# ==========================================
if not df_client.empty:
    try:
        plt.figure(figsize=(8, 6))
        
        df_client['Char_Count'] = df_client['Phrase'].apply(lambda x: len(str(x)))
        
        x = df_client['Char_Count']
        y = df_client['Total_Latency_ms']
        
        plt.scatter(x, y, color='purple', alpha=0.6)
        
        # Trend line
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), "r--", alpha=0.8, label="Trend")

        plt.xlabel('Input Length (Characters)')
        plt.ylabel('Time to First Audio (ms)')
        plt.title('Fig 4: Scalability Test')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        
        plt.savefig(f"{OUTPUT_FOLDER}/chart_4_scalability.png")
        print("generated Chart 4...")
    except Exception as e:
        print(f"❌ Error generating Chart 4: {e}")

print(f"\n✅ Done! Check the '{OUTPUT_FOLDER}' folder for your images.")