import random
import pandas as pd


SINGLE_GENRE_SUMMARY = pd.read_json("word_dicts/genre_single_summary.json")

def generate_genre_summary(genres: list[str]) -> str:
    summary = []
    for genre in genres:
        if genre in SINGLE_GENRE_SUMMARY.columns:
            summary.append(f"{genre} : {random.choice(SINGLE_GENRE_SUMMARY[genre].values)}")

    return "\n".join(summary)





