from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent

text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )

raw_documents = PyPDFLoader(file_path ='./italy_travel.pdf').lazy_load()
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings())

print(db)

print("-"*100)  


tool = create_retriever_tool(
    db.as_retriever(),
    "italy_travel",
    "Searches and returns documents regarding Italy."
)

tools = [tool]

print(tools)

print("-"*100)  


memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

#print(memory)

#print("-"*100)  

llm = ChatOpenAI(temperature = 0)

agent_executor = create_conversational_retrieval_agent(llm, tools, memory_key='chat_history', verbose=True)

print("-"*100)  

#print(agent_executor)  


#for msg in result['chat_history']:
#    if hasattr(msg, 'answer'):
#        print(msg.content)
# 

#agent_executor({"input": "Tell me something about Pantheon"})    

result = agent_executor({"input": "what can I visit in India in 3 days?"})

print(result['output'])