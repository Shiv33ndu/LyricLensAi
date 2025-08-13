from utils.features import Features
from pathlib import Path

_MODEL_PATH = Path(__file__).parent/ "model"/ "genre_predict_145_prob.pkl"
_VECTORIZER_PATH = Path(__file__).parent/ "model"/ "vectorizer_145_prob.pkl"
_ENCODER_PATH = Path(__file__).parent/ "model"/ "encoder_145_prob.pkl"


ob = Features(_MODEL_PATH, _VECTORIZER_PATH, _ENCODER_PATH)

result = ob.predict_genre("")

print(result)
print("\n\n\n\n\n")
# print(result['genres'])

# print(result['summary'])

# print(result['triggers'])