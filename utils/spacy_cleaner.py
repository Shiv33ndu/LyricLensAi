from .nlp_resources import nlp

def spacy_cleaner(lyrics: str) -> str:
    """
    This cleaner is used to remove the verbs from the lyrics so that we can simply extract 
    only the vocabs that decide the mood of the song, so that we can create a dictionary 
    of vocabs genre-wise, that will be used later on to suggest words to 
    make a lyrics sound like from a specific genre

    """

    string = nlp(lyrics)
    token = []

    for w in string:
        if w.is_stop:   # if word is stopword skip and move to next word 
            continue
        if w.is_punct or w.is_space: #skip if word is from punc or is space
            continue
        if w.pos_ not in ['NOUN', 'PROPN', 'ADJ']: # skip if word is not these 
            continue

        token.append(w.lemma_.lower())
    
    return " ".join(token) 