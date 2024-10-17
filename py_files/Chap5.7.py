import os

from langchain_community.document_loaders import TextLoader

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

from langchain_openai import OpenAI

os.environ["OPENAI_API_KEY"]

raw_documents = TextLoader('dialogue.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0, separator = "\n",)
documents = text_splitter.split_documents(raw_documents)

db = FAISS.from_documents(documents, OpenAIEmbeddings())

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

llm=OpenAI()
retriever = db.as_retriever()

combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

results = rag_chain.invoke({"input": "What is the reason for calling?"})

answer_text = results['answer']

print(f"Answer : {answer_text}")


