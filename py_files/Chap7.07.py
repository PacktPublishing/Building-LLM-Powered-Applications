import os
import lancedb

from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings

from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature = 0)

embeddings = OpenAIEmbeddings()

uri = "data/sample-lancedb"
db = lancedb.connect(uri)

table_name = "movies"
table = db.open_table(table_name)

docsearch = LanceDB(connection = db, table_name=table_name, embedding=embeddings)

retriever = docsearch.as_retriever(return_source_documents=True)

tool = create_retriever_tool(
    retriever,
    "movies",
    "Searches and returns recommendations about movies."
)
tools = [tool]

agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)

result = agent_executor.invoke({"input": "suggest me some action movies"})

print(result['output'])

