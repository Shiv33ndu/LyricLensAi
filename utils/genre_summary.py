import random
import pandas as pd
from pathlib import Path 


_JSON_PATH = Path(__file__).parent/"word_dicts"/"genre_single_summary.json"
SINGLE_GENRE_SUMMARY = pd.read_json(_JSON_PATH)

def generate_genre_summary(genres: list[str]) -> str:
    summary = []
    for genre in genres:
        if genre in SINGLE_GENRE_SUMMARY.columns:
            summary.append(f"{genre} : {random.choice(SINGLE_GENRE_SUMMARY[genre].values)}")

    return "\n".join(summary)