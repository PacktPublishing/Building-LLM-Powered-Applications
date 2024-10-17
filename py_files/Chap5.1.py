
from langchain_openai import OpenAI

import os

os.environ["OPENAI_API_KEY"]

llm = OpenAI()
print(llm('tell me a joke'))

