import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Load Data
csv_file = "benchmark_results.csv"
if not os.path.exists(csv_file):
    print("❌ No data found! Run the app and speak a few phrases first.")
    exit()

df = pd.read_csv(csv_file)

# Cleanup: Truncate long phrases for the chart labels
df['Short_Phrase'] = df['Phrase'].apply(lambda x: x[:15] + "..." if len(x) > 15 else x)

# --- CHART 1: Latency Waterfall (Stacked Bar) ---
plt.figure(figsize=(10, 6))

# Plot Transcription (Bottom)
p1 = plt.bar(df.index, df['Step_1_Transcription_ms'], label='Local STT (Client)', color='#4da6ff')

# Plot Server Wait (Top)
p2 = plt.bar(df.index, df['Step_2_Server_Wait_ms'], bottom=df['Step_1_Transcription_ms'], label='Network + Cloud AI', color='#ff9933')

plt.xlabel('Test Phrase ID')
plt.ylabel('Latency (ms)')
plt.title('End-to-End Latency Breakdown (Hybrid Architecture)')
plt.xticks(df.index, df['Short_Phrase'], rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add Total Labels on top
for i, total in enumerate(df['Total_Latency_ms']):
    plt.text(i, total + 20, f"{int(total)}ms", ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("chart_latency_waterfall.png") # Saves the image for your report
plt.show()

# --- STATISTICS SUMMARY ---
avg_total = df['Total_Latency_ms'].mean()
avg_stt = df['Step_1_Transcription_ms'].mean()
avg_server = df['Step_2_Server_Wait_ms'].mean()

print("-" * 40)
print(f"📊 ANALYTICS SUMMARY")
print("-" * 40)
print(f"Total Samples:      {len(df)}")
print(f"Avg Total Latency:  {avg_total:.2f} ms")
print(f"  ├─ Avg Local STT: {avg_stt:.2f} ms ({(avg_stt/avg_total)*100:.1f}%)")
print(f"  └─ Avg Cloud Wait: {avg_server:.2f} ms ({(avg_server/avg_total)*100:.1f}%)")
print("-" * 40)
print("✅ Charts saved to folder!")