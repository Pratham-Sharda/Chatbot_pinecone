from main import ChatBot
import streamlit as st

bot = ChatBot()
    
st.set_page_config(page_title="Envio Chatbot")
with st.sidebar:
    st.title('Envio Chatbot')

def extract_answer(text):
    # Define the marker to split the text
    marker = "Answer:"
    
    # Find the index of the marker
    marker_index = text.find(marker)
    
    # If the marker is found, extract the text after the marker
    if marker_index != -1:
        # Extract the text after the marker
        answer = text[marker_index + len(marker):].strip()
        return answer
    else:
        # If the marker is not found, return an appropriate message
        return "No answer found."
# Function for generating LLM response
def generate_response(input):
    result = bot.rag_chain.invoke(input)
    return extract_answer(result)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, let's answer your question "}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Getting your answer "):
            response = generate_response(input) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)