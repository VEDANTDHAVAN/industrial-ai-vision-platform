import pandas as pd
from pathlib import Path

LOG_FILE = Path("logs/predictions.csv")

df = pd.read_csv(LOG_FILE)

avg_conf = df["confidence"].mean()

manual_rate = (
    (df["prediction"] == "MANUAL_REVIEW").mean()
) * 100

print(f"Average Confidence: {avg_conf:.4f}")
print(f"Manual Review Rate: {manual_rate:.2f}%")

if avg_conf < 0.80:
    print("\nALERT: Confidence drop detected")

if manual_rate > 30:
    print("\nALERT: Excessive manual reviews")

if avg_conf < 0.80 and manual_rate > 30:
    print("\nRETRAINING RECOMMENDED")