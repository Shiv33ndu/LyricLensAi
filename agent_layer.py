from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from utils.features import Features  # your Feature class
from pathlib import Path
from agent_direction import qna_temp, classifier_temp, suggestion_temp, review_temp, chit_chat_temp
import streamlit as st

# === Paths ===
_MODEL_PATH = Path(__file__).parent / "model" / "genre_predict_145_prob.pkl"
_VECTORIZER_PATH = Path(__file__).parent / "model" / "vectorizer_145_prob.pkl"
_ENCODER_PATH = Path(__file__).parent / "model" / "encoder_145_prob.pkl"

# === Init Feature class ===
features = Features(_MODEL_PATH, _VECTORIZER_PATH, _ENCODER_PATH)

# === Main LLM for Q&A ===
qa_model = OllamaLLM(model="llama3.2")

qa_template = qna_temp()
qa_prompt = ChatPromptTemplate.from_template(qa_template)
qa_chain = qa_prompt | qa_model



# === Classifier LLM ===
classifier_model = OllamaLLM(model="llama3.2", temperature=0)

classifier_template = classifier_temp()
classifier_prompt = ChatPromptTemplate.from_template(classifier_template)
classifier_chain = classifier_prompt | classifier_model


# === Chit-Chat ===
chitchat_model = OllamaLLM(model="llama3.2", temperature=2)
chitchat_template = chit_chat_temp()
chitchat_prompt = ChatPromptTemplate.from_template(chitchat_template)
chitchat_chain = chitchat_prompt | chitchat_model


# === Suggestion LLM for genre word recommendations ===
suggestion_model = OllamaLLM(model="llama3.2", temperature=1)

suggestion_template = suggestion_temp()
suggestion_prompt = ChatPromptTemplate.from_template(suggestion_template)
suggestion_chain = suggestion_prompt | suggestion_model


# === Review Model LLM for giving a quick review on a given lyrics ===
review_model = OllamaLLM(model="llama3.2")

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
    
    prediction_markdown = build_prediction_table(genres, summary, triggers_line)

    base_text = (
        f"🎵 Genres of your lyrics: {genres}\n"
        f"📝 Summary: {summary}\n"
        f"💡 Trigger words for {result['genres'][0]}: {triggers_line}\n"
    )

    print(base_text)  # for base text result cross check console print and debug 

    review_content = {"prediction_output": prediction_markdown, "genre": top_genre, "words": ", ".join(trigger_words), "lyrics": lyrics}

    return review_content 


# def handle_input(user_input: str) -> str:
#     lyric = user_input
#     classifications = ""
#     for raw_cls in classifier_chain.stream({"user_input": user_input}):
#         classifications += raw_cls

#     classification = classifications.strip().lower()
    
#     if classification == "lyrics":
#         result = features.predict_genre(lyric)
#         base_reply = pretty_print_prediction(lyric, result, ask_for_words=False)

#         # Ask the user if they want to adapt lyrics to a genre
#         print("\nAgent:", base_reply)
#         follow_up = input("\nDo you want to make it sound more like a specific genre? (yes/no): ").strip().lower()
#         if follow_up in {"yes", "y"}:
#             genre_choice = input("Which genre? ").strip()
#             suggested_words = features.give_words_to_agent(genre_choice)
#             for chunk in suggestion_chain.stream({"genre": genre_choice, "words": ", ".join(suggested_words), "lyrics": lyric}):
#                 print(chunk, end='', flush=True)
            
#             return "" 
#         else:
#             return base_reply
#     else:
#         chat_history_text = render_chat_history()
#         for answer in qa_chain.stream({"chat_history": chat_history_text, "question": user_input}):
#             print(answer, end='', flush=True)
#         return ""

# def check(lyrics: str):
#     result = features.predict_genre(lyrics)
#     print(f"Result= {result}")

def build_prediction_table(genres, summary, triggers_line) -> str:
    header = "| **Genre** | **Summary** | **Trigger Words** |\n"
    header += "|-----------|-------------|-------------------|\n"

    # If genres can be multiple, join them nicely
    genre_str = ", ".join(genres) if isinstance(genres, (list, tuple)) else str(genres)

    return header + f"| {genre_str} | {summary} | {triggers_line} |"



def handle_input_ui(user_input: str):
    lyric = user_input
    classifications = ""

    for raw_cls in classifier_chain.stream({"user_input": user_input}):
        classifications += raw_cls
    
    classification = classifications.strip().lower()

    if classification == 'lyrics':
        result = features.predict_genre(lyric)
        
        review_content = pretty_print_prediction(lyric, result, ask_for_words=False)
        
        print(review_content['prediction_output'])  # for debug and cross checking
        
        for chunk in review_chain.stream(review_content):
            yield chunk

    elif classification == "question":
        chat_history_text = render_chat_history()

        for chunk in qa_chain.stream({"chat_history": chat_history_text, "question": user_input}):
            yield chunk

    elif classification == "chat":
        chat_history_text = render_chat_history()

        for chunk in chitchat_chain.stream({"chat_history": chat_history_text, "user_input": user_input}):
            yield chunk
    
    elif classification == "suggestion":
        chat_history_text = render_chat_history()
        
        genre_choice = input("Which genre? ").strip()
        suggested_words = features.give_words_to_agent(genre_choice)
        
        for chunk in suggestion_chain.stream({"chat_history": chat_history_text, "genre": genre_choice, "words": ", ".join(suggested_words), "lyrics": lyric}):
            yield chunk 

    else:
        yield "Hmm, I’m not sure what that was 🤔. Try pasting lyrics or asking me about music!"






# if __name__ == "__main__":
#     print("🎶 Lyrics Genre Assistant — paste lyrics or ask a question. (q to quit)")
#     while True:
#         print("\n" + "-" * 30)
#         user_input = input("You: ").strip()
#         if user_input.lower() in {"q", "quit", "exit"}:
#             break

#         reply = handle_input(user_input)
#         CHAT_MEMORY.append((user_input, reply))
#         print("Agent:", reply)
# print(check('I tried so and got so'))