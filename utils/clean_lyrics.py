from nltk.tokenize import TweetTokenizer
from .nlp_resources import nlp, stop_words

def clean_lyrics(lyrics: str) -> str:
    if lyrics == None:
        return ""
    
    # to lowercase the tokens, and to preserve the slangs
    token = TweetTokenizer().tokenize(lyrics.lower())  

    # remove non-alphabetical strings
    words = [w for w in token if w.isalpha()]

    # only return the tokens that are not stopwords
    return " ".join([w for w in words if w not in stop_words])