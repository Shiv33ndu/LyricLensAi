import nltk
import spacy

#NLTK coropora - download once, quietly
for res in ("stopwords", "punkt"):
    nltk.download(res, quiet=True)

#spaCy English model - load once
nlp = spacy.load("en_core_web_sm")
stop_words = set(nltk.corpus.stopwords.words("english"))