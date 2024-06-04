# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Pinecone
# from langchain.llms import HuggingFaceHub
# from pinecone import Pinecone as PineconeClient, ServerlessSpec
# from dotenv import load_dotenv
# import os
# from langchain.document_loaders import UnstructuredURLLoader
# def format_hell(docms):
#     return "\n\n".join(docul.page_content for docul in docms)
# class ChatBot():
#     load_dotenv()

#     embeddings = HuggingFaceEmbeddings()

#     pinecone_api_key = os.getenv('PINECONE_API_KEY')
#     pinecone = PineconeClient(api_key=pinecone_api_key)

#     index_name = "langchain-demo-100"

#     # if index_name in pinecone.list_indexes().names():  
#     #     pinecone.delete_index(index_name)  

#     if index_name not in pinecone.list_indexes().names():
#         pinecone.create_index(
#             name=index_name, 
#             dimension=768, 
#             metric='cosine', 
#             spec=ServerlessSpec(
#                 cloud='aws', 
#                 region='us-east-1'
#             )
#         )
#         docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
#     else:
#         docsearch = Pinecone.from_existing_index(index_name, embeddings)

#     repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#     llm = HuggingFaceHub(
#     repo_id=repo_id, 
#     model_kwargs={"temperature": 0.8, "top_k": 50}, 
#     huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_TOKEN')
#     )

#     from langchain import PromptTemplate

#     template = """
#     You are an environmentalist. Answer the following question about the environment using the given context.

#     - For general questions, provide concise answers.
#     - If the question asks for elaboration or details, give a detailed and expanded answer.
#     - If the question requires listing items, answer in bullet points.

#     Context: {context}
#     Question: {question}
#     Answer:
#     """

#     prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    
#     from langchain.schema.runnable import RunnablePassthrough
#     from langchain.schema.output_parser import StrOutputParser

#     from langchain_core.runnables import RunnableParallel



#     rag_chain_from_docs = (
#         RunnablePassthrough.assign(context=(lambda x: format_hell(x["context"])))
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     rag_chain_with_source = RunnableParallel(
#         {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
#     ).assign(answer=rag_chain_from_docs)

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv
import os
from langchain.document_loaders import UnstructuredURLLoader
def format_hell(docms):
    return "\n\n".join(docul.page_content for docul in docms)
class ChatBot():
    load_dotenv()

    embeddings = HuggingFaceEmbeddings()

    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone = PineconeClient(api_key=pinecone_api_key)

    index_name = "langchain-demo-100"

    # if index_name in pinecone.list_indexes().names():  
    #     pinecone.delete_index(index_name)  

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

    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceHub(
    repo_id=repo_id, 
    model_kwargs={"temperature": 0.95, "top_k":15,'max_length': 5000}, 
    huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_TOKEN')
    )

    from langchain import PromptTemplate
    template = """
    You are an environmental specialist. Your task is to provide a comprehensive and detailed long answer to the following question based on the provided context. Ensure that your response is well-structured and covers all relevant aspects.

    When the question involves listing or mentioning different steps, reasons, methods, or procedures, provide the answers in bullet points (as shown below), ensuring that all relevant points are included:
    - Reason 1
    - Reason 2
    - Reason 3
    - Reason 4
    and so on...

    For all other types of questions, provide your response in detailed paragraph format.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser

    from langchain_core.runnables import RunnableParallel



    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_hell(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)











