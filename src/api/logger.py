import csv
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR/"predictions.csv"
print("CSV PATH:", LOG_FILE.resolve())

def log_prediction(
    filename: str,
    prediction: str,
    confidence: float,
    ok_probability: float,
    defect_probability: float,
    model_version: str,
):
    print("log_prediction called")

    file_exists = LOG_FILE.exists()

    try:
        with open(
            LOG_FILE,
            mode="a",
            newline="",
            encoding="utf-8"
        ) as f:

            print("file opened")

            writer = csv.writer(f)

            row = [
                datetime.now().isoformat(),
                filename,
                prediction,
                confidence,
                ok_probability,
                defect_probability,
                model_version
            ]

            print("writing row:", row)

            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "filename",
                    "prediction",
                    "confidence",
                    "ok_probability",
                    "defect_probability",
                    "model_version"
                ])

            writer.writerow(row)

            f.flush()

            print("row written")

    except Exception as e:
        print("CSV ERROR:", e)