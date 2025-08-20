# 🎵 LyricLensAI  
*LyricLensAi is a creative writing assistant that analyzes lyrics, identifies genre influences, provides feedback, and offers suggestions to adapt lyrics to different genres.*


[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/) 
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/) 
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E.svg)](https://scikit-learn.org/stable/)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) 
[![Status](https://img.shields.io/badge/Project-Active-success.svg)]()  

---

##  Features

- **Genre Detection**: A logistic regression model analyzes input lyrics and predicts genre influences.
- **Explainability**: Summaries and trigger words explain why certain genres were detected.
- **Creative Feedback**: Gets lyrical suggestions and reviews via Llama 3.2 Instruct (via Hugging Face).
- **Multi-Mode Agent**: Supports modes like Q&A, chit-chat, lyrical review, and genre suggestion.
- **English-only Filter**: Detects if lyrics are not in English and prompts the user accordingly.

---

##  Tech Stack

- Python 3.10+  
- **ML Components**:
  - Logistic Regression (sklearn)
  - Text cleaning (NLTK, spaCy)
  - Language detection (`langdetect`)
- **LLM Integration**:
  - Local: Ollama + Llama3.2  
  - Remote: Hugging Face inference (`meta-llama/Llama-3.2-3B-Instruct`) using `langchain-huggingface`
- **UI**: Streamlit for front-end chat interface
- **Prompt Chain**: LangChain for orchestrating prompts, templates, and memory

---

##  Repository Structure

LyricLensAi/
├── model/ # Pretrained genre model artifacts
│ ├── genre_predict_145_prob.pkl
│ ├── vectorizer_145_prob.pkl
│ └── encoder_145_prob.pkl
├── utils/ # Utility routines (text cleaning, feature extraction)
├── agent_direction.py # Custom conversation templates
├── agent_layer.py # Orchestrates prompt chains and model integration
├── streamlit_app.py # Entrypoint for Streamlit UI
├── requirements.txt # Project dependencies
├── README.md # Project documentation
└── LICENSE # MIT License

---

---

##  Installation & Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/Shiv33ndu/LyricLensAi.git
   cd LyricLensAi

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt

4. (Optional) TO use Hugging Face LLaMA model:
   - Sign up at Hugging Face and generate an API Token.
   - Store it as an environment variabe:
     ```bash
     export HUGGINGFACEHUB_API_TOKEN="your_token_here"

---

## Running Locally

   ```bash
   streamlit run streamlit_app.py

Interact with the UI:
- Paste your lyrics (English Only)
- Ask questions ("What's the genre? Review my lyrics.")
- Request suggestions ("Make this sound more like Country or Hip-Hop")

---


