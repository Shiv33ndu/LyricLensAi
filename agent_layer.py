# import os
# import sys
# import signal
# from pathlib import Path
# from utils.features import Features



# from langchain_ollama import ChatOllama
# from langchain_core.tools import tool
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langgraph.prebuilt import create_react_agent
# from langgraph.checkpoint.memory import MemorySaver


# # ---------- Config ----------
# MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3")  # change if you prefer another local model
# _MODEL_PATH = Path(__file__).parent/ "model"/ "genre_predict_145_prob.pkl"
# _VECTORIZER_PATH = Path(__file__).parent/ "model"/ "vectorizer_145_prob.pkl"
# _ENCODER_PATH = Path(__file__).parent/ "model"/ "encoder_145_prob.pkl"

# # Load your classical ML stack once
# features_obj = Features(
#     model_path=_MODEL_PATH,
#     vectorizer_path=_VECTORIZER_PATH,
#     encoder_path=_ENCODER_PATH,
# )

# # Keep lightweight session state (handy for follow-ups / debugging)
# SESSION_STATE = {
#     "last_lyrics": None,
#     "last_genres": None,
#     "last_top_genre": None,
#     "last_triggers": None,
# }


# # ---- Tools ----
# @tool
# def predict_genres_and_triggers(lyrics: str) -> str:
#     """Given song lyrics, predicts top 5 genres and trigger words for the top genre."""
#     result = features_obj.predict_genre(lyrics)
#     genres = result["genres"]
#     summary = result["summary"]
#     triggers = result["triggers"]

#     if not genres:
#         return "I couldn't detect any genres from the lyrics."

#     SESSION_STATE["last_lyrics"] = lyrics
#     SESSION_STATE["last_genres"] = genres
#     SESSION_STATE["last_top_genre"] = genres[0]
#     SESSION_STATE["last_triggers"] = triggers

#     genres_line = ", ".join(genres)
#     triggers_line = ", ".join(triggers) if triggers else "—"

#     return (
#         f"🎵 Top Genres: {genres_line}\n"
#         f"📝 Summary: {summary}\n"
#         f"💡 Trigger Words for {genres[0]}: {triggers_line}\n\n"
#         f"Would you like to make your lyrics sound like a specific genre?"
#     )


# @tool
# def suggest_words_for_genre(genre: str) -> str:
#     """Suggests helpful words for a given genre."""
#     available_cols = list(features_obj.word_dictionary_of_genres.columns)
#     lookup = {c.lower(): c for c in available_cols}
#     key = lookup.get(genre.strip().lower())
#     if not key:
#         return f"Unknown genre. Available: {', '.join(sorted(available_cols)[:10])}..."
#     suggestions = features_obj.suggestions(key)["suggestions"]
#     words_line = ", ".join(sorted(suggestions))
#     return f"✨ To give your lyrics a {key} vibe, use words like: {words_line}"


# @tool
# def recall_last_prediction(_: str) -> str:
#     """Returns the last predicted genres, top genre, and triggers."""
#     genres = SESSION_STATE.get("last_genres")
#     if not genres:
#         return "No previous prediction found."
#     top_genre = SESSION_STATE.get("last_top_genre")
#     triggers = SESSION_STATE.get("last_triggers")
#     triggers_line = ", ".join(triggers) if triggers else "—"
#     return (
#         f"Last prediction:\n"
#         f"- Genres: {', '.join(genres)}\n"
#         f"- Top genre: {top_genre}\n"
#         f"- Trigger words: {triggers_line}"
#     )


# # ---- Build LangGraph ReAct Agent ----
# def build_agent():
#     llm = ChatOllama(model="llama3", temperature=0.3)
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 "You are a lyrics genre prediction assistant. "
#                 "If user gives lyrics, call predict_genres_and_triggers. "
#                 "If user asks to adapt lyrics to a genre, call suggest_words_for_genre. "
#                 "If user refers to previous result, call recall_last_prediction."
#             ),
#             MessagesPlaceholder("messages"),
#         ]
#     )

#     memory = MemorySaver()
#     tools = [predict_genres_and_triggers, suggest_words_for_genre, recall_last_prediction]
#     return create_react_agent(llm, tools, checkpointer=memory, prompt=prompt)


# # ---- Main ----
# def main():
#     agent = build_agent()
#     config = {"configurable": {"thread_id": "lyrics_session"}}

#     print("🎶 Lyrics Genre Assistant (LangGraph) 🎶")
#     while True:
#         user_input = input("\nYou: ")
#         if user_input.lower() in {"exit", "quit"}:
#             break
#         events = agent.stream({"messages": [("user", user_input)]}, config)
#         for event in events:
#             if "agent" in event:
#                 print("Agent:", event["agent"]["messages"][-1].content)


# if __name__ == "__main__":
#     main()






from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from utils.features import Features  # your Feature class
from pathlib import Path

# === Paths ===
_MODEL_PATH = Path(__file__).parent / "model" / "genre_predict_145_prob.pkl"
_VECTORIZER_PATH = Path(__file__).parent / "model" / "vectorizer_145_prob.pkl"
_ENCODER_PATH = Path(__file__).parent / "model" / "encoder_145_prob.pkl"

# === Init Feature class ===
features = Features(_MODEL_PATH, _VECTORIZER_PATH, _ENCODER_PATH)

# === Main LLM for Q&A ===
qa_model = OllamaLLM(model="llama3.2")

qa_template = """
You are an AI assistant for a music genre classification tool.
Use the conversation history to keep context when answering questions.

Conversation history (most recent last):
{chat_history}

If the user asks about lyrics prediction, ask them to paste lyrics.
Otherwise, answer their general music/genre questions concisely.

Question: {question}
"""
qa_prompt = ChatPromptTemplate.from_template(qa_template)
qa_chain = qa_prompt | qa_model

# === Classifier LLM ===
classifier_model = OllamaLLM(model="llama3.2", temperature=0)

classifier_template = """
Classify the following user input as either exactly "lyrics" or "question".
- If it appears to be song lyrics/poetry/multi-line creative text → "lyrics"
- If it's a factual request or explanation → "question"
Only output one lowercase word: lyrics or question.

User input:
{user_input}
"""
classifier_prompt = ChatPromptTemplate.from_template(classifier_template)
classifier_chain = classifier_prompt | classifier_model

# === Suggestion LLM for genre word recommendations ===
suggestion_model = OllamaLLM(model="llama3.2")

suggestion_template = """
You are a creative lyrics assistant.
The genre is: {genre}
Here is a list of few suggested words for this genre: {words}
User's given {lyrics} can incorporate on these suggested {words} for the his desired {genre}  

Write a short, friendly suggestion for the user on how they could incorporate the vibe of this genre into their lyrics.
Tell them they don't have to use the exact words—synonyms or related terms are fine.

Example:
"To give a hip-hop street vibe, you can use words like street, hustle, musclecars—you don't have to use them exactly, just pick synonyms or phrases that match the feel."

Now create the suggestion for the given genre and words.
"""
suggestion_prompt = ChatPromptTemplate.from_template(suggestion_template)
suggestion_chain = suggestion_prompt | suggestion_model



review_model = OllamaLLM(model="llama3.2")

review_template = """
You are a creative lyrics assistance who has been an award winning song-writer and creative writer.
The genre is: {genre}
The {words} that triggered this {genre}

Write a constructive review on the current {lyrics} given by the user, and the trigger {words} used by user. Also keep the tone friendly and motivating.

Example: 
"The words you have used in this lyrics really give it the mood you are looking for, you can enhance this more by using some more creative words like denial for no, using metaphors would help as well "
"""
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
        lines.append(f"User: {u_short}\nAI: {a_short}")
    return "\n---\n".join(lines)

# def pretty_print_prediction(result: dict) -> str:
#     if not result or not result.get("genres"):
#         return "I couldn't detect any genres—please paste some lyrics."
    
#     genres = ", ".join(result["genres"])
#     summary = result.get("summary", "")
#     triggers = result.get("triggers", [])
#     triggers_line = ", ".join(triggers) if triggers else "—"

#     # Automatically add genre word suggestions for the top genre
#     top_genre = result["genres"][0]
#     suggested_words = features.give_words_to_agent(top_genre)
#     suggestion_text = suggestion_chain.invoke({"genre": top_genre, "words": ", ".join(suggested_words)})

#     return (
#         f"🎵 Top Genres: {genres}\n"
#         f"📝 Summary: {summary}\n"
#         f"💡 Trigger words for {top_genre}: {triggers_line}\n"
#         f"{suggestion_text}"
#     )

# def handle_input(user_input: str) -> str:
#     raw_cls = classifier_chain.invoke({"user_input": user_input})
#     classification = str(raw_cls).strip().lower()
    
#     if classification == "lyrics":
#         result = features.predict_genre(user_input)
#         return pretty_print_prediction(result)
#     else:
#         chat_history_text = render_chat_history()
#         answer = qa_chain.invoke({"chat_history": chat_history_text, "question": user_input})
#         return str(answer)


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
    review_text = review_chain.invoke({"genre": top_genre, "words": ", ".join(trigger_words), "lyrics" : lyrics})

    base_text = (
        f"🎵 Top Genres: {genres}\n"
        f"📝 Summary: {summary}\n"
        f"💡 Trigger words for {result['genres'][0]}: {triggers_line}\n"
        f"{review_text}\n"
    )



    # Only suggest words if user asked for them
    if ask_for_words:
        top_genre = result["genres"][0]
        suggested_words = features.give_words_to_agent(top_genre)
        suggestion_text = suggestion_chain.invoke({"genre": top_genre, "words": ", ".join(suggested_words)})
        base_text += suggestion_text

    return base_text


def handle_input(user_input: str) -> str:
    lyric = user_input
    raw_cls = classifier_chain.invoke({"user_input": user_input})
    classification = str(raw_cls).strip().lower()
    
    if classification == "lyrics":
        result = features.predict_genre(lyric)
        base_reply = pretty_print_prediction(lyric, result, ask_for_words=False)

        # Ask the user if they want to adapt lyrics to a genre
        print("\nAgent:", base_reply)
        follow_up = input("\nDo you want to make it sound more like a specific genre? (yes/no): ").strip().lower()
        if follow_up in {"yes", "y"}:
            genre_choice = input("Which genre? ").strip()
            suggested_words = features.give_words_to_agent(genre_choice)
            suggestion_text = suggestion_chain.invoke({"genre": genre_choice, "words": ", ".join(suggested_words), "lyrics": lyric})
            return suggestion_text #base_reply + "\n" + suggestion_text
        else:
            return base_reply
    else:
        chat_history_text = render_chat_history()
        answer = qa_chain.invoke({"chat_history": chat_history_text, "question": user_input})
        return str(answer)


if __name__ == "__main__":
    print("🎶 Lyrics Genre Assistant — paste lyrics or ask a question. (q to quit)")
    while True:
        print("\n" + "-" * 30)
        user_input = input("You: ").strip()
        if user_input.lower() in {"q", "quit", "exit"}:
            break

        reply = handle_input(user_input)
        CHAT_MEMORY.append((user_input, reply))
        print("Agent:", reply)
