import streamlit as st
from agent_layer import handle_input_ui, CHAT_MEMORY, render_chat_history
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import time

# about page and heading description
st.set_page_config(page_title='LyricLens AI', layout='centered')


# --- Typewriter Animation ---
def typewriter(text, speed=0.05):
    container = st.empty()
    typed_text = ""
    for char in text:
        typed_text += char
        container.markdown(f"<h1 style='text-align: center; color: transparent; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); background-clip: text;'>{typed_text}</h1>", unsafe_allow_html=True)
        time.sleep(speed)

# --- Subheading Typewriter ---
def typewriter_subtitle(text, speed=0.05):
    container = st.empty()
    typed_text = ""
    for char in text:
        typed_text += char
        container.markdown(f"<h4 style='text-align: center; color: white;'>{typed_text}</h4>", unsafe_allow_html=True)
        time.sleep(speed)

# --- App Start ---
if "initialized" not in st.session_state:
    typewriter("LyricLens.Ai", speed=0.08)
    typewriter_subtitle("Your Creative Writing Companion!", speed=0.05)

    # --- Fade In Markdown (CSS animation) ---
    st.markdown(
        """
        <style>
        .fade-in {
            animation: fadeIn 2s ease-in forwards;
            opacity: 0;
        }
        .fade-out {
        animation: fadeOut .2s ease-out forwards;
        opacity: 1;
        }


        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        
        @keyframes fadeOut {
        from {opacity: 1;}
        to {opacity: 0;}
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="fade-in">

        🎵 **What I Am**

        I'm a ML model that can:  
        - Classify your lyrics and tell you how they *feel* genre-wise.  
        - Suggest the words/phrases to match the style you want (Pop, Rock, Hip-Hop, etc.).  
        - Help polish your hooks, verses, and choruses with creative suggestions.  
        - And I can chat and discuss about creative songwriting in general.  

        ---

        👉 **How to Get Started**  
        1. Type or paste your lyrics in the chat box.  
        2. Or Ask: *"Suggest some line for my chorus."*  
        3. Get instant insights + lyric suggestions! 😎  

        💡 *Tip: The more lyrics you share, the better my suggestions get.*  

        </div>
        """,
        unsafe_allow_html=True
    )


# now chat's will be stateless per session.
if 'initialized' not in st.session_state:
    st.session_state.clear()
    st.session_state.initialized = True
# st.session_state.clear() # clearing leftover states


def update_history(user_msg: str, assistant_msg: str):
    # Update Streamlit memory
    st.session_state.messages.append(AIMessage(content=assistant_msg))
    # st.session_state.messages.append(HumanMessage(content=user_msg))

    # Update agent memory
    CHAT_MEMORY.append((user_msg, assistant_msg))

# initialize the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(f"<pre style='font-family: Pacifico, cursive; font-size:18px; white-space:pre-wrap;'>{message.content}</pre>",
        unsafe_allow_html=True)
            # st.markdown(f"```\n{message.content}\n```")
    
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


# message input box 
prompt = st.chat_input('Paste you lyrics or ask me question about writing..')

# if a user submit a prompt
if prompt:

    # add the prompt on the screen and also store into HumanMessage
    with st.chat_message("user"):
        st.markdown(f"```\n{prompt}\n```")  # swill show on the screen

        st.session_state.messages.append(HumanMessage(prompt)) # will add the same message to session of streamlit for history recall and chat memory
    

    with st.chat_message('assistant'):
        placeholder = st.empty()
        response = ""
        
        for chunk in handle_input_ui(prompt):  # streamed output from agent
            response += chunk
            placeholder.markdown(response + "▌")
        placeholder.markdown(response)
        update_history(prompt, response) # updating chat history for CHAT_MEMORY and streamlit_session 
    
    