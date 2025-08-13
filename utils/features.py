"""
The model has two basic features
- To give words suggestion based on selected genre by the user
- To list out those words that might have triggered the genre 
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import random
from sklearn.base import BaseEstimator, TransformerMixin
from utils.genre_summary import generate_genre_summary
from utils.clean_lyrics import clean_lyrics
from utils.spacy_cleaner import spacy_cleaner

class Features:
    """
    All the features of the models like genre prediction, trigger words, suggestion words are coded as methods that can be called, once the path is given as param at the time of Class Object creation 
    """
    
    def __init__(self, 
                 model_path: str | Path,
                 vectorizer_path: str | Path,
                 encoder_path: str | Path
                 ):
        
        self.model: BaseEstimator = joblib.load(model_path) # loading the model 
        self.vectorizer: TransformerMixin = joblib.load(vectorizer_path) # loading the vectorizer
        self.encoder: BaseEstimator = joblib.load(encoder_path) # loading the encoder
        
        # dictionary will be loaded when the Feature class object is created
        _JSON_PATH = Path(__file__).parent/ "word_dicts" / "genre_top_100_words.json"
        self.word_dictionary_of_genres = pd.read_json(_JSON_PATH) 



    # suggestion feature 
    def suggestions(self, genre_name: str) -> set:
        """
        This method will suggest words to users if they pass the genre as param they want to write their lyrics for.

        this simply looks for top words from the dictionary of genre created from dataset and randomly picks 5 words and returns them
        """


        # extract the column and values with the desired genre name from the dict
        words = self.word_dictionary_of_genres[genre_name].values

        # then loop through the values and randomly pick 5 words out of them to return   
        return {"suggestions" : set([random.choice(words) for _ in range(6)])}




    # trigger words feature
    def trigger_words(self, lyrics: str, genre: str) -> list[str]:
        """
        This feature will return the trigger words for top genre

        Top genre is the genre with highest probability assigend by the model

        lyrics : string | Takes the lyrics as string to work on
        genre : string  | Takes the top genre name as string to give words for 
        """
        if lyrics == "":
            return []
        
        clean_lyrics = set(spacy_cleaner(lyrics).split())
    
        all_feature_names = self.vectorizer.get_feature_names_out() # we took out all the feature names out 
        all_genre_classes = list(self.encoder.classes_)
        genre_index = all_genre_classes.index(genre) # we need the index number of the genres
        genre_coeff = self.model.coef_[genre_index]    # we need coefficients of the desied genre using index

        # we will sort the genre_coeff from high to low value wise and taking indexes
        top_coeff = genre_coeff.argsort()[::-1]

        global_trigger_words = [all_feature_names[i] for i in top_coeff]   

        lyric_specific_words = list(set(global_trigger_words) & clean_lyrics) # we only take out the words that are in lyrics

        return list(set([random.choice(lyric_specific_words) for _ in range(1, 5)]))





    def predict_genre(self, lyrics: str):
        """
        This is to predict the probability of genre using the model, it takes two parameters

        lyrics : string | on which is the prediction will occur
        """


        if lyrics == "":
            return {"genres" : [], "summary" : "", "triggers" : []}
        
        clean_lrc = clean_lyrics(lyrics)
        print(clean_lrc[:20])

        vector = self.vectorizer.transform([clean_lrc])
        probable = self.model.predict_proba(vector)[0]

        top_indices = np.argsort(probable)[::-1][:5]

        genres = [self.encoder.inverse_transform([i])[0] for i in top_indices]   

        summary = generate_genre_summary(genres)  # we are making genre summary as per the list of the genre model predicted 

        trg_words = self.trigger_words(clean_lrc, genres[0])  # we also return the trigger words for the top genre

        return {"genres" : genres, "summary" : summary, "triggers" : trg_words}

