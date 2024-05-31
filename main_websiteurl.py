# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Pinecone
# from langchain.llms import HuggingFaceHub
# from pinecone import Pinecone as PineconeClient, ServerlessSpec
# from dotenv import load_dotenv
# import os
# import requests
# from bs4 import BeautifulSoup
# from langchain.document_loaders import UnstructuredURLLoader

# def fetch_sitemap(url):
#     response = requests.get(url)
#     response.raise_for_status()
#     return response.content

# def parse_sitemap(xml_content):
#     soup = BeautifulSoup(xml_content, 'lxml-xml')
#     sitemap_urls = [loc.text for loc in soup.find_all('loc')]
#     return sitemap_urls

# def get_all_urls(sitemap_url):
#     all_urls = []
#     urls_to_process = [sitemap_url]

#     while urls_to_process:
#         current_url = urls_to_process.pop()
#         content = fetch_sitemap(current_url)
#         urls = parse_sitemap(content)

#         for url in urls:
#             if url.endswith('.xml'):
#                 urls_to_process.append(url)
#             else:
#                 all_urls.append(url)

#     return all_urls



# class ChatBot():
#     load_dotenv()
#     # ############################
#     # # URL of the initial sitemap
#     # initial_sitemap_url = 'https://urbanemissions.info/sitemap.xml'

#     # # Get all URLs
#     # all_extracted_urls = get_all_urls(initial_sitemap_url)

#     # # html_extensions = ('.html', '.htm', '/')
#     # # non_html_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp',
#     # #                    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
#     # #                    '.mp4', '.avi', '.mov', '.wmv', '.flv', '.mp3', '.wav', '.ogg',
#     # #                    '.zip', '.rar', '.tar', '.gz', '.7z', '.xml', '.txt', '.json', '.csv')

#     # html_urls = [url for url in all_extracted_urls if url.endswith(('.html', '.htm', '/')) and not url.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp',
#     #                    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
#     #                    '.mp4', '.avi', '.mov', '.wmv', '.flv', '.mp3', '.wav', '.ogg',
#     #                    '.zip', '.rar', '.tar', '.gz', '.7z', '.xml', '.txt', '.json', '.csv'))]

#     # ############################
    
#     # loader = TextLoader('./daata2.txt')
#     # documents = loader.load()
#     URL_L=['https://urbanemissions.info/','https://urbanemissions.info/blog-pieces/whats-polluting-delhis-air/','https://urbanemissions.info/india-apna/'
#            'https://urbanemissions.info/tools/','https://urbanemissions.info/tools/vapis/','https://urbanemissions.info/tools/aqi-calculator','https://urbanemissions.info/tools/atmos',
#            'https://urbanemissions.info/tools/wrf-configuration-notes/','https://urbanemissions.info/india-air-quality/india-ncap-review/#sim-40-2021',
#            'https://urbanemissions.info/blog-pieces/airquality-smog-towers-crazy/','https://urbanemissions.info/blog-pieces/delhi-diwali-2017-understanding-emission-loads/',
#            'https://urbanemissions.info/blog-pieces/resources-how-to-access-aqdata-in-india/','https://urbanemissions.info/india-emissions-inventory/emissions-in-india-open-agricultural-forest-fires/']
#     loader = UnstructuredURLLoader(urls=URL_L)
#     documents = loader.load()

#     text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=20)
#     docs = text_splitter.split_documents(documents)

#     embeddings = HuggingFaceEmbeddings()

#     pinecone_api_key = os.getenv('PINECONE_API_KEY')
#     pinecone = PineconeClient(api_key=pinecone_api_key)

#     index_name = "langchain-demo-2"

#     if index_name in pinecone.list_indexes().names():  
#         pinecone.delete_index(index_name)  

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
#     # else:
#     #     docsearch = Pinecone.from_existing_index(index_name, embeddings)

#     huggingface_api_token = os.getenv('HUGGINGFACE_API_KEY')


#     repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#     llm = HuggingFaceHub(
#     repo_id=repo_id, 
#     model_kwargs={"temperature": 0.8, "top_k": 50}, 
#     huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_TOKEN')
#     )

#     from langchain import PromptTemplate

#     template = """
#     You are a environmentalist. These Human will ask you a questions about environment. Use following piece of context to answer the question. 
#     You answer with atleast 5 sentences.

#     Context: {context}
#     Question: {question}
#     Answer: 
#     """

#     prompt = PromptTemplate(template=template, input_variables=["context", "question"])

#     from langchain.schema.runnable import RunnablePassthrough
#     from langchain.schema.output_parser import StrOutputParser

#     rag_chain = (
#         {"context": docsearch.as_retriever(), "question": RunnablePassthrough()} 
#         | prompt 
#         | llm 
#         | StrOutputParser()
#     )



# bot = ChatBot()
# input = input("Ask me anything: ")
# result = bot.rag_chain.invoke(input)
# print(result) 




from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv
import os
from langchain.document_loaders import UnstructuredURLLoader

class ChatBot():
    load_dotenv()
    URL_L=['https://urbanemissions.info/','https://urbanemissions.info/blog-pieces/whats-polluting-delhis-air/','https://urbanemissions.info/india-apna/'
            'https://urbanemissions.info/tools/','https://urbanemissions.info/tools/vapis/','https://urbanemissions.info/tools/aqi-calculator','https://urbanemissions.info/tools/atmos',
            'https://urbanemissions.info/tools/wrf-configuration-notes/','https://urbanemissions.info/india-air-quality/india-ncap-review/#sim-40-2021',
            'https://urbanemissions.info/blog-pieces/airquality-smog-towers-crazy/','https://urbanemissions.info/blog-pieces/delhi-diwali-2017-understanding-emission-loads/',
            'https://urbanemissions.info/blog-pieces/resources-how-to-access-aqdata-in-india/','https://urbanemissions.info/india-emissions-inventory/emissions-in-india-open-agricultural-forest-fires/']
    loader = UnstructuredURLLoader(urls=URL_L)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone = PineconeClient(api_key=pinecone_api_key)

    index_name = "langchain-demo-2"

    if index_name in pinecone.list_indexes().names():  
        pinecone.delete_index(index_name)  

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
    # else:
    #     docsearch = Pinecone.from_existing_index(index_name, embeddings)

    huggingface_api_token = os.getenv('HUGGINGFACE_API_KEY')


    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceHub(
    repo_id=repo_id, 
    model_kwargs={"temperature": 0.8, "top_k": 50}, 
    huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_TOKEN')
    )

    from langchain import PromptTemplate

    template = """
    You are a data expert in field of text data analysis. These Human will ask you a questions about environment. Use following piece of context to answer the question. 
    You answer with precision and don't unnecessarily expand the answer unless required.


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
input = input("Ask me anything: ")
result = bot.rag_chain.invoke(input)
print(result) 
