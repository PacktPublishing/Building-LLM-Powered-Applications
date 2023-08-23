
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.memory import ConversationBufferMemory 
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain import SerpAPIWrapper
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import BaseTool, Tool, tool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain import PromptTemplate, LLMChain

#from langchain import HuggingFaceHub
st.set_page_config(page_title="GlobeBotter", page_icon="🌐")
st.header('🌐 Welcome to Globebotter, your travel assistant with Internet access. What are you planning for your next trip?')



load_dotenv()

#os.environ["HUGGINGFACEHUB_API_TOKEN"]
openai_api_key = os.environ['OPENAI_API_KEY']
serpapi_api_key = os.environ['SERPAPI_API_KEY']

search = SerpAPIWrapper()
text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )

raw_documents = PyPDFLoader('italy_travel.pdf').load()
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings())

memory = ConversationBufferMemory(
    return_messages=True, 
    memory_key="chat_history", 
    output_key="output"
)

llm = ChatOpenAI()
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

agent = create_conversational_retrieval_agent(llm, tools, memory_key='chat_history', verbose=True)

user_query = st.text_input(
    "**Where are you planning your next vacation?**",
    placeholder="Ask me anything!"
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "memory" not in st.session_state:
    st.session_state['memory'] = memory


for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


def display_msg(msg, author):

    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

if user_query:
    display_msg(user_query, 'user')
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)

if st.sidebar.button("Reset chat history"):
    st.session_state.messages = []





