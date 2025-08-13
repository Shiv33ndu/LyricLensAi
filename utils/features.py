"""
The model has two basic features
- To give words suggestion based on selected genre by the user
- To list out those words that might have triggered the genre 
"""
import pandas as pd
from pathlib import Path
import joblib
import random
from sklearn.base import BaseEstimator, TransformerMixin

class Features:
    """
    All the features of the models like genre prediction, trigger words, suggestion words are coded as methods that can be called, once the path is given as param at the time of Class Object creation 
    """
    
    def __init__(self, 
                 model_path: str | Path,
                 vectorizer_path: str | Path,
                 encoder_path: str | Path):
        
        self.model: BaseEstimator = joblib.load(model_path) # loading the model 
        self.vectorizer: TransformerMixin = joblib.load(vectorizer_path) # loading the vectorizer
        self.encoder: BaseEstimator = joblib.load(encoder_path) # loading the encoder
        
        self.word_dictionary_of_genres = pd.read_json("word_dicts/genre_top_100_words.json") # dictionary will be loaded when the Feature class object is created



    # suggestion feature 
    def suggestions(self, genre_name: str) -> set:
        """
        This method will suggest words to users if they pass the genre as param they want to write their lyrics for.

        this simply looks for top words from the dictionary of genre created from dataset and randomly picks 5 words and returns them
        """


        # extract the column and values with the desired genre name from the dict
        words = self.word_dictionary_of_genres[genre_name].values

        # then loop through the values and randomly pick 5 words out of them to return   
        return set([random.choice(words) for _ in range(6)])




    # trigger words feature
    def trigger_words(self, lyrics: str, genre: str) -> list[str]:
        """
        This feature will return the trigger words for top genre

        Top genre is the genre with highest probability assigend by the model

        lyrics : string | Takes the lyrics as string to work on
        genre : string  | Takes the top genre name as string to give words for 
        """

        





    def predict_genre(lyrics: str, model: BaseEstimator, vectorizer: TransformerMixin, encoder: BaseEstimator):
        pass