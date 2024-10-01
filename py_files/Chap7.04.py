import os
import lancedb

from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings

# RetrievalQA
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

embeddings = OpenAIEmbeddings()

uri = "data/sample-lancedb"
db = lancedb.connect(uri)

table = "movies"

db.open_table(table)

docsearch = LanceDB(connection = db, table_name=table, embedding=embeddings)

query =  "I'm looking for an animated action movie. What could you suggest to me?"

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)

query = "I'm looking for an animated action movie. What could you suggest to me?"

result = qa({"query": query})

print(result['result']+"\r\n")

print(result['source_documents'][0])

