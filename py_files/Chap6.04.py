import os
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.utilities    import SerpAPIWrapper

from langchain.tools import Tool

from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent


load_dotenv()
os.environ["SERPAPI_API_KEY"]
search = SerpAPIWrapper()

text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )

raw_documents = PyPDFLoader(file_path ='./italy_travel.pdf').lazy_load()
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings())

llm = ChatOpenAI(temperature = 0)

tools = [
    Tool.from_function(
        func=search.run,
        name="Search",
        description="useful for when you need to answer questions about current events"
    ),
    create_retriever_tool(
        db.as_retriever(),
        "italy_travel",
        "Searches and returns documents regarding Italy."
    )
    ]
agent_executor = create_conversational_retrieval_agent(llm, tools, memory_key='chat_history', verbose=True)

agent_executor({"input": "what can I visit in India in 3 days?"})

agent_executor({"input": "What is the weather currently in Delhi?"})


