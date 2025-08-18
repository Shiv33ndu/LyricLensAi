from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from utils.features import Features  # your Feature class
from utils.utils import is_english
from pathlib import Path
from agent_direction import qna_temp, classifier_temp, suggestion_temp, review_temp, chit_chat_temp
import streamlit as st
import json


from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from dotenv import load_dotenv
import os

#load variables from env into OS enviornment
load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# === Paths ===
_MODEL_PATH = Path(__file__).parent / "model" / "genre_predict_145_prob.pkl"
_VECTORIZER_PATH = Path(__file__).parent / "model" / "vectorizer_145_prob.pkl"
_ENCODER_PATH = Path(__file__).parent / "model" / "encoder_145_prob.pkl"

# === Init Feature class ===
features = Features(_MODEL_PATH, _VECTORIZER_PATH, _ENCODER_PATH)

# === Main LLM for Q&A ===

#qa_model = OllamaLLM(model="llama3.2")  # uncomment this if using local model, instead of llama3.2 use your model name

qa_hf_endpt = HuggingFaceEndpoint(
                                repo_id="meta-llama/Llama-3.2-3B-Instruct",
                                task="conversational",
                                max_new_tokens=512,
                                temperature=0
                              )  # for using API called models, from huggingface repo

qa_model = ChatHuggingFace(llm=qa_hf_endpt)

qa_template = qna_temp()
qa_prompt = ChatPromptTemplate.from_template(qa_template)
qa_chain = qa_prompt | qa_model






# === Classifier LLM ===
# classifier_model = OllamaLLM(model="llama3.2", temperature=0) # uncomment this when using local model on your own machine

classify_hf_endpt = HuggingFaceEndpoint(
                                repo_id="meta-llama/Llama-3.2-3B-Instruct",
                                task="conversational",
                                max_new_tokens=512,
                                temperature=0
                              )  

classifier_model = ChatHuggingFace(llm=classify_hf_endpt)

classifier_template = classifier_temp()
classifier_prompt = ChatPromptTemplate.from_template(classifier_template)
classifier_chain = classifier_prompt | classifier_model


# === Chit-Chat ===
# chitchat_model = OllamaLLM(model="llama3.2", temperature=2)  #uncomment to use your own local model
chitchat_hf_endpt = HuggingFaceEndpoint(
                                repo_id="meta-llama/Llama-3.2-3B-Instruct",
                                task="conversational",
                                max_new_tokens=512,
                                temperature=2
                              )  

chitchat_model = ChatHuggingFace(llm=chitchat_hf_endpt)

chitchat_template = chit_chat_temp()
chitchat_prompt = ChatPromptTemplate.from_template(chitchat_template)
chitchat_chain = chitchat_prompt | chitchat_model


# === Suggestion LLM for genre word recommendations ===
# suggestion_model = OllamaLLM(model="llama3.2", temperature=1)    # uncomment to use your own model
suggest_hf_endpt = HuggingFaceEndpoint(
                                repo_id="meta-llama/Llama-3.2-3B-Instruct",
                                task="conversational",
                                max_new_tokens=512,
                                temperature=1
                              )  

suggestion_model = ChatHuggingFace(llm=suggest_hf_endpt)


suggestion_template = suggestion_temp()
suggestion_prompt = ChatPromptTemplate.from_template(suggestion_template)
suggestion_chain = suggestion_prompt | suggestion_model


# === Review Model LLM for giving a quick review on a given lyrics ===
# review_model = OllamaLLM(model="llama3.2")     # uncomment to use your own model

rvw_hf_endpt = HuggingFaceEndpoint(
                                repo_id="meta-llama/Llama-3.2-3B-Instruct",
                                task="conversational",
                                max_new_tokens=512,
                                temperature=1.5
                              )  

review_model = ChatHuggingFace(llm=rvw_hf_endpt)

review_template = review_temp()
review_prompt = ChatPromptTemplate.from_template(review_template)
review_chain = review_prompt | review_model


# === Memory ===
CHAT_MEMORY: list[tuple[str, str]] = []  # [(user_msg, assistant_msg), ...]

def render_chat_history(max_turns: int = 8) -> str:
    turns = CHAT_MEMORY[-max_turns:]
    lines = []
    for u, a in turns:
        u_short = (u[:500] + " ...") if len(u) > 500 else u
        a_short = (a[:500] + " ...") if len(a) > 500 else a
        lines.append(f"user: {u_short}\nassistant: {a_short}")
    return "\n---\n".join(lines)


def pretty_print_prediction(lyrics: str, result: dict, ask_for_words: bool = False) -> str:
    if not result or not result.get("genres"):
        return "I couldn't detect any genres—please paste some lyrics."
    
    genres = ", ".join(result["genres"])
    summary = result.get("summary", "")
    triggers = result.get("triggers", [])
    triggers_line = ", ".join(triggers) if triggers else "—"

    # Automatically add genre word suggestions for the top genre
    top_genre = result["genres"][0]
    trigger_words = features.trigger_words(lyrics, top_genre)

    base_text = (
        f"🎵 Genres of your lyrics: {genres}\n"
        f"📝 Summary: {summary}"
        f"💡 Trigger words for {result['genres'][0]}: {triggers_line}\n"
    )

    print(base_text)  # for base text result cross check console print and debug 

    review_content = {
                    "all_genres": genres,   # all genres, whose lyrcal elements are present
                    "summary": summary,     # summary, for each genres' 
                    "trigger_words": ", ".join(trigger_words), # trigger words for the genres that were predicted
                    "top_genre": top_genre, # the genre, the given lyric heavily sounds like 
                    "lyrics": lyrics        # shared lyrics by the user 
                    }

    return review_content 


def handle_input_ui(user_input: str):
    lyric = user_input
    classifications = ""

    for raw_cls in classifier_chain.stream({"user_input": user_input}):
        classifications += raw_cls.content
    
    # classification = classifications.strip().lower()
    classification = json.loads(classifications)
    print(classification)

    if classification.get('category') == 'lyrics':
        
        # very first check for language of the lyric, as model is english only
        if not is_english(lyric):
            yield "Right now I can only analyze English lyrics. Please provide your lyrics in English."
            return 
        
        result = features.predict_genre(lyric)
        
        review_content = pretty_print_prediction(lyric, result, ask_for_words=False)
        
        for chunk in review_chain.stream({
                    "all_genres": review_content["all_genres"],   # all genres, whose lyrcal elements are present
                    "summary": review_content["summary"],
                    "genre_1": review_content["summary"][0],     # summary, for each genres' 
                    "genre_2": review_content["summary"][1],     # summary, for each genres' 
                    "genre_3": review_content["summary"][2],     # summary, for each genres' 
                    "genre_4": review_content["summary"][3],     # summary, for each genres' 
                    "genre_5": review_content["summary"][4],     # summary, for each genres' 
                    "trigger_words": review_content["trigger_words"], # trigger words for the genres that were predicted
                    "top_genre": review_content["top_genre"], # the genre, the given lyric heavily sounds like 
                    "lyrics": review_content["lyrics"]        # shared lyrics by the user 
                    }):
            yield chunk.content

    elif classification.get('category') == "question":
        chat_history_text = render_chat_history()

        for chunk in qa_chain.stream({"chat_history": chat_history_text, "question": user_input}):
            yield chunk.content

    elif classification.get('category') == "chat":
        chat_history_text = render_chat_history()

        for chunk in chitchat_chain.stream({"chat_history": chat_history_text, "user_input": user_input}):
            yield chunk.content
    
    elif classification.get('category') == "suggestion":
        chat_history_text = render_chat_history()
        
        genre_choice = classification.get('genre')
        suggested_words = features.give_words_to_agent(genre_choice)
        
        for chunk in suggestion_chain.stream({"chat_history": chat_history_text, "genre": genre_choice, "words": ", ".join(suggested_words), "lyrics": lyric}):
            yield chunk.content 

    else:
        yield "Hmm, I’m not sure what that was 🤔. Try pasting lyrics or asking me about music!"



