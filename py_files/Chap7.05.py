import os
import lancedb
import pandas as pd

from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings

# RetrievalQA
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

embeddings = OpenAIEmbeddings()

uri = "data/sample-lancedb"
db = lancedb.connect(uri)

table_name = "movies"

table = db.open_table(table_name)
# Convert the table to a Pandas DataFrame
md = pd.DataFrame(table.to_pandas())
#print(md)

docsearch = LanceDB(connection = db, table_name=table_name, embedding=embeddings)

df_filtered = md[md['genres'].apply(lambda x: 'Comedy' in x)]

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={'data': df_filtered}), return_source_documents=True)

query = "I'm looking for a movie with animals and an adventurous plot."
result = qa({"query": query})

print(result['result']+"\r\n")

#print(result['source_documents'][0])

