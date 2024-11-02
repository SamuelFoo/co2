from pathlib import Path

PRESENTATION_MEDIA_DIR = Path("presentation_media")
DATA_DIR = Path("data/co2_readings/processed")
PRESENTATION_MEDIA_DIR.mkdir(exist_ok=True)
CO2_DATABASE_PATH = "co2_readings.db"
