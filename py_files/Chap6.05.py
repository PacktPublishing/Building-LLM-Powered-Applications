import streamlit as st

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

from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamlitCallbackHandler


st.set_page_config(page_title="GlobeBotter", page_icon="")
st.header('🌐  Welcome to Globebotter, your travel assistant with Internet access. What are you planning for your next trip?')

st.balloons()

load_dotenv()
os.environ["SERPAPI_API_KEY"]
search = SerpAPIWrapper()

text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )

raw_documents = PyPDFLoader(file_path ='/home/pvandervoort/Study/Packt/italy_travel.pdf').lazy_load()
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings())

memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

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

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent_executor(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
#        st.write(response)
        st.write(response['output'])

if st.sidebar.button("Reset chat history"):
    st.session_state.messages = []
    