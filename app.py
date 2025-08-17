import streamlit as st
from agent_layer import handle_input_ui, CHAT_MEMORY, render_chat_history
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# about page and heading description
st.set_page_config(page_title='LyricLens AI - Your creative writing assistant', layout='centered')
st.title('LyricLensAi')
st.subheader("Your Creative Writing Assistant!")
st.write('\n\nHi there!! I am your creative writing assistant! Want me analyze you lyrics?\n\n Tell you how your song sounds lyrically, or you need help on completing your catchy hooks! All that assistance is just one chat away!! 😎')

# st.session_state.clear()


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
            st.markdown(f"```\n{message.content}\n```")
    
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="assets\Music.gif"):
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