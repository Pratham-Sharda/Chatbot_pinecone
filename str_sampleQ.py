import streamlit as st
from main_websiteurl import ChatBot

bot = ChatBot()

st.set_page_config(page_title="Envio Chatbot")
with st.sidebar:
    st.title('Envio Chatbot')

    # Sample questions
    sample_questions = [
        "What Do You understand by Apna Program",
        "What is polluting Delhi's air",
        "Atmos is used for what purpose and also menton its latest version?",
        "Tell me about Vapis",
        "Are smog towers effective in reducing air pollution?",
        "What is climate change?",
        "What was the main goal of NCAP",
        "What are dust storms",
        "DEscribe the steps used in apna program",
        "elaborate on sim air family of tools?",
        "Custom question"
    ]

    # Dropdown for sample questions
    question_type = st.selectbox("Choose a question or enter your own:", sample_questions)

    if question_type == "Custom question":
        input = st.text_input("Enter your custom question:")
    else:
        input = question_type

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
    
def complete_sentence(text):
    # Ensure the text ends with a complete sentence
    end_punctuation = ['.', '!', '?']
    if text[-1] not in end_punctuation:
        last_punct_index = max(text.rfind(p) for p in end_punctuation)
        if last_punct_index != -1:
            return text[:last_punct_index + 1]
    return text

# Function for generating LLM response
def generate_response(input):
    result = bot.rag_chain_with_source.invoke(input)
    raw_answer = extract_answer(result['answer'])
    answer = complete_sentence(raw_answer)  # Ensure the answer is complete
    sources = [doc.metadata['source'] for doc in result['context']]
    return answer, sources


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, let's answer your question "}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if input:
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Getting your answer"):
                answer, sources = generate_response(input)
                st.write(answer)
                if sources:
                    st.write("Sources:")
                    for source in sources:
                        st.write(f"- {source}")
        message = {"role": "assistant", "content": answer}
        st.session_state.messages.append(message)
