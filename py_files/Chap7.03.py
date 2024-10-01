import os
import openai
import lancedb

from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings

openai.api_key = os.environ["OPENAI_API_KEY"]

embeddings = OpenAIEmbeddings()

uri = "data/sample-lancedb"
db = lancedb.connect(uri)

table = "movies"

db.open_table(table)

#print(db[table].head())

docsearch = LanceDB(connection = db, table_name=table, embedding=embeddings)

query =  "I'm looking for an animated action movie. What could you suggest to me?"

docs = docsearch.similarity_search(query)

for doc in docs:
    print( doc.page_content + "\r\n" ) 



