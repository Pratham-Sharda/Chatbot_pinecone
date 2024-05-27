from main import ChatBot
import streamlit as st

# Create an instance of ChatBot
bot = ChatBot()

# Set Streamlit page configuration
st.set_page_config(page_title="Envio Chatbot")

# Define the sidebar title
with st.sidebar:
    st.title('Envio Chatbot')

# Function to extract answer from the response text
def extract_answer(text):
    marker = "Answer:"
    marker_index = text.find(marker)
    if marker_index != -1:
        answer = text[marker_index + len(marker):].strip()
        return answer
    else:
        return "No answer found."

# Function to generate LLM response
def generate_response(input):
    result = bot.rag_chain.invoke(input)
    return extract_answer(result)

# Initialize chat messages in session state
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, let's answer your question "}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
user_input = st.chat_input("Ask me anything:")

# Handle user input and generate response
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Getting your answer "):
                response = generate_response(user_input) 
                st.write(response) 
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
