import random
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords

SINGLE_GENRE_SUMMARY = pd.read_json("word_dicts/genre_single_summary.json")

def generate_genre_summary(genres: list[str]) -> str:
    summary = []
    for genre in genres:
        if genre in SINGLE_GENRE_SUMMARY.columns:
            summary.append(f"{genre} : {random.choice(SINGLE_GENRE_SUMMARY[genre].values)}")

    return "\n".join(summary)


def spacy_cleaner(lyrics: str) -> str:
    pass


def clean_lyrics(lyrics: str) -> str:
    if lyrics == None:
        return ""
    token = TweetTokenizer().tokenize(lyrics.lower())

    words = [w for w in token if w.isalpha()]

    stop_words = set(stopwords.words("english"))

    return " ".join([w for w in words if w not in stop_words])