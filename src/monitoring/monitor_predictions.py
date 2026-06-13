from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

LOG_FILE = Path("logs/predictions.csv")

def load_data():
    if not LOG_FILE.exists():
        raise FileNotFoundError(
            f"Log file not found: {LOG_FILE}"
        )
    
    return pd.read_csv(LOG_FILE)

def show_summary(df):
    print("\n=== MONITORING REPORT ===\n")
    print("Total Predictions:")
    print(len(df))

    print("\nPrediction Counts:")
    print(df["prediction"].value_counts())

    print("\nAverage Confidence:")
    print(round(df["confidence"].mean(), 4))

def plot_prediction_distribution(df):
    counts = df["prediction"].value_counts()

    plt.figure(figsize=(6,4))

    counts.plot(kind="bar")

    plt.title("Prediction Distribution")
    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig("outputs/prediction_distribution.png", dpi=300)

    plt.show()

def plt_confidence(df):
    plt.figure(figsize=(6,4))

    plt.hist(
        df["confidence"], bins=10
    )

    plt.title("Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig("outputs/confidence_distribution.png", dpi=300)
    plt.show()

def main():
    df = load_data()
    show_summary(df)

    plot_prediction_distribution(df)
    plt_confidence(df)

if __name__ == "__main__":
    main()