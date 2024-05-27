from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv
import os

class ChatBot():
    load_dotenv()
    loader = TextLoader('./daata2.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone = PineconeClient(api_key=pinecone_api_key)

    index_name = "langchain-demo-2"

    if index_name not in pinecone.list_indexes().names():
        pinecone.create_index(
            name=index_name, 
            dimension=768, 
            metric='cosine', 
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            )
        )
        docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    else:
        docsearch = Pinecone.from_existing_index(index_name, embeddings)

    huggingface_api_token = os.getenv('HUGGINGFACE_API_KEY')


    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceHub(
    repo_id=repo_id, 
    model_kwargs={"temperature": 0.8, "top_k": 5}, 
    huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_TOKEN')
    )

    from langchain import PromptTemplate

    template = """
    You are a environmentalist. These Human will ask you a questions about environment. Use following piece of context to answer the question. 
    You answer with atleast 5 sentences.

    Context: {context}
    Question: {question}
    Answer: 
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser

    rag_chain = (
        {"context": docsearch.as_retriever(), "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )



bot = ChatBot()
u_input = input("Ask me anything: ")
result = bot.rag_chain.invoke(u_input)
print(result) 
