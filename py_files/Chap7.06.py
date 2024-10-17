import os
import lancedb

#import pandas as pd
#import numpy as np

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
docsearch = LanceDB(connection = db, table_name=table_name, embedding=embeddings)

"""
# Convert the table to a Pandas DataFrame
# check if weighted_rate is well defined as Float32 based on error from langchain
md = pd.DataFrame(table.to_pandas())
print(md.dtypes)

# Attempt to convert all values to floats and check for invalid entries
md['weighted_rate'] = pd.to_numeric(md['weighted_rate'], errors='coerce')

# Check if there are any NaN values, indicating invalid or non-float data
invalid_entries = md[md['weighted_rate'].isna()]

if invalid_entries.empty:
    print("All values are valid floats.")
else:
    print("Invalid entries found:")
    print(invalid_entries)

# Check for NaN values
nan_values = md['weighted_rate'].isna().sum()
print(f"Number of NaN values: {nan_values}")

# Check for infinite values
inf_values = np.isinf(md['weighted_rate']).sum()
print(f"Number of infinite values: {inf_values}")
"""



#qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff",
#    retriever=docsearch.as_retriever(search_kwargs={'filter': {weighted_rate__gt:7}}), return_source_documents=True)

# I've tried a lot of solutions, at least this one works
# found based on https://github.com/langchain-ai/langchain/issues/9195

where_filter = {"where filter":"weighted_rate", "operator": "GreaterThanEqual", "valueFloat": 7}

qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), 
        chain_type="stuff",
        retriever = docsearch.as_retriever(
            search_kwargs={"where filter": where_filter},
            return_source_documents=True)
)


retriever_kwargs = {"where_filter": where_filter}

query = "I'm looking for a movie with animals and an adventurous plot."
result = qa({"query": query})

print("\r\n"+result['result']+"\r\n")



