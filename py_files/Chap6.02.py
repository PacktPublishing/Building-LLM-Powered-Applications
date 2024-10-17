from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
raw_documents = PyPDFLoader(file_path ='./italy_travel.pdf').lazy_load()
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings())
memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
llm = ChatOpenAI()

qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=db.as_retriever(), memory=memory, verbose=False)

result = qa_chain.invoke({'question':'Give me some review about the Pantheon'})

print(result)

print('*' * 100)

print(result['answer'])

#for msg in result['chat_history']:
#    if hasattr(msg, 'answer'):
#        print(msg.content)    


