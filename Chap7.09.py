import os
import lancedb

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_community.vectorstores import LanceDB


embeddings = OpenAIEmbeddings()
uri = "data/sample-lancedb"
db = lancedb.connect(uri)
table_name = "movies"
table = db.open_table(table_name)

docsearch = LanceDB(connection = db, table_name=table_name, embedding=embeddings)


template = """You are a movie recommender system that help users to find movies that match their preferences.
Use the following pieces of context to answer the question at the end.
For each question, suggest three movies, with a short description of the plot and the reason why the user migth like it.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Your response:"""
 
PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(llm=OpenAI(),
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs)

query = "I'm looking for a funny action movie, any suggestion?"

result = qa.invoke({'query':query})
print(result['result'])