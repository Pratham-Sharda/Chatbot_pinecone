# from main_single_url import ChatBot
# import streamlit as st

# # Create an instance of ChatBot
# bot = ChatBot()

# # Set Streamlit page configuration
# st.set_page_config(page_title="Envio Chatbot")

# # Define the sidebar title
# with st.sidebar:
#     st.title('Envio Chatbot')

# # Function to extract answer from the response text
# def extract_answer(text):
#     marker = "Answer:"
#     marker_index = text.find(marker)
#     if marker_index != -1:
#         answer = text[marker_index + len(marker):].strip()
#         return answer
#     else:
#         return "No answer found."

# # Function to generate LLM response
# def generate_response(input):
#     result = bot.rag_chain.invoke(input)
#     return extract_answer(result)

# # Initialize chat messages in session state
# if "messages" not in st.session_state.keys():
#     st.session_state.messages = [{"role": "assistant", "content": "Welcome, let's answer your question "}]

# # Display chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# # User-provided prompt
# user_input = st.chat_input("Ask me anything:")

# # Handle user input and generate response
# if user_input:
#     st.session_state.messages.append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.write(user_input)

#     # Generate response if last message is not from assistant
#     if st.session_state.messages[-1]["role"] != "assistant":
#         with st.chat_message("assistant"):
#             with st.spinner("Getting your answer "):
#                 response = generate_response(user_input) 
#                 st.write(response) 
#         message = {"role": "assistant", "content": response}
#         st.session_state.messages.append(message)

# import streamlit as st
# from vector_saved import ChatBot

# bot = ChatBot()
# def truncate_to_punctuation(s):
#     # Check if the string ends with a full stop, comma, colon, semicolon, or hyphen
#     if s.endswith(('.', ',', ':', ';', '-')):
#         return s
    
#     # Find the nearest punctuation from the end
#     punctuations = ['.', ',', ':', ';', '-']
#     nearest_punctuation = -1
#     for p in punctuations:
#         pos = s.rfind(p)
#         if pos > nearest_punctuation:
#             nearest_punctuation = pos
    
#     # If no punctuation is found, return an empty string
#     if nearest_punctuation == -1:
#         return ''
    
#     # Return the string up to the nearest punctuation
#     a=s[:nearest_punctuation + 1]
#     a=a[:-1]
#     a=a+"."
#     return a
# st.set_page_config(page_title="Envio Chatbot")
# with st.sidebar:
#     st.title('Envio Chatbot')

#     # Sample questions
#     sample_questions = ["Custom question",
#         "What Do You understand by Apna Program",
#         "What is polluting Delhi's air",
#         "Atmos is used for what purpose ?",
#         "Tell me about Vapis",
#         "Are smog towers effective in reducing air pollution?",
#         "What is climate change?",
#         "What was the main goal of NCAP",
#         "What are dust storms",
#         "DEscribe the steps used in apna program",
#         "elaborate on sim air family of tools?"
        
#     ]

#     # Dropdown for sample questions
#     question_type = st.selectbox("Choose a question or enter your own:", sample_questions)

#     if question_type == "Custom question":
#         input = st.text_input("Enter your custom question:")
#     else:
#         input = question_type

# def extract_answer(text):
#     # Define the marker to split the text
#     marker = "Answer:"
    
#     # Find the index of the marker
#     marker_index = text.find(marker)
    
#     # If the marker is found, extract the text after the marker
#     if marker_index != -1:
#         # Extract the text after the marker
#         answer = text[marker_index + len(marker):].strip()
#         return answer
#     else:
#         # If the marker is not found, return an appropriate message
#         return "No answer found."
    
# # def complete_sentence(text):
# #     # Ensure the text ends with a complete sentence
# #     end_punctuation = ['.', '!', '?']
# #     if text[-1] not in end_punctuation:
# #         last_punct_index = max(text.rfind(p) for p in end_punctuation)
# #         if last_punct_index != -1:
# #             return text[:last_punct_index + 1]
# #     return text

# # Function for generating LLM response
# def generate_response(input):
#     result = bot.rag_chain_with_source.invoke(input)
#     # raw_answer = truncate_to_punctuation(extract_answer(result['answer']))
#     # answer = complete_sentence(raw_answer)  # Ensure the answer is complete
#     sources = [doc.metadata['url'] for doc in result['context']]
#     return result['answer'], sources


# # Store LLM generated responses
# if "messages" not in st.session_state.keys():
#     st.session_state.messages = [{"role": "assistant", "content": "Welcome, let's answer your question "}]

# # Display chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# if input:
#     st.session_state.messages.append({"role": "user", "content": input})
#     with st.chat_message("user"):
#         st.write(input)

#     # Generate a new response if last message is not from assistant
#     if st.session_state.messages[-1]["role"] != "assistant":
#         with st.chat_message("assistant"):
#             with st.spinner("Getting your answer"):
#                 answer, sources = generate_response(input)
#                 st.write(answer)
#                 if sources:
#                     st.write("Sources:")
#                     for source in sources:
#                         st.write(f"- {source}")
#         message = {"role": "assistant", "content": answer}
#         st.session_state.messages.append(message)


import streamlit as st
from vector_saved import ChatBot

# Initialize the ChatBot instance
bot = ChatBot()

# Function for generating LLM response
def generate_response(input, history):
    # Ensure the history is passed correctly
    result = bot.ask(input, history)
    
    # Extract the answer from the result
    answer = result['answer']
    
    # Extract sources if available
    sources = [doc.metadata['url'] for doc in result['context']]
    return answer, sources

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, let's answer your question "}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

with st.sidebar:
    st.title('Envio Chatbot')

    # Sample questions
    sample_questions = ["Custom question",
        "What Do You understand by Apna Program",
        "What is polluting Delhi's air",
        "Atmos is used for what purpose?",
        "Tell me about Vapis",
        "Are smog towers effective in reducing air pollution?",
        "What is climate change?",
        "What was the main goal of NCAP",
        "What are dust storms",
        "Describe the steps used in Apna Program",
        "Elaborate on Sim Air family of tools?"
    ]

    # Dropdown for sample questions
    question_type = st.selectbox("Choose a question or enter your own:", sample_questions)

    if question_type == "Custom question":
        input = st.text_input("Enter your custom question:")
    else:
        input = question_type

if input:
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Getting your answer"):
                # Retrieve the conversation history
                history = st.session_state.messages

                # Generate the response using the chatbot
                try:
                    answer, sources = generate_response(input, history)
                   
                except Exception as e:
                    st.write(f"Error: {e}")
                else:
                    st.write(answer)
                    if sources:
                        st.write("Sources:")
                        for source in sources:
                            st.write(f"- {source}")

        # Append the assistant's response to the session state messages
        st.session_state.messages.append({"role": "assistant", "content": answer})
       

