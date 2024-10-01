
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain

template = """Sentence: {sentence} Translation in {language}:"""
prompt = PromptTemplate(template=template, input_variables=["sentence", "language"])

os.environ["OPENAI_API_KEY"]

llm = OpenAI(temperature=0)

#llm_chain = LLMChain(prompt=prompt, llm=llm, output_parser=StrOutputParser())
#he class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0. Use RunnableSequence, e.g., `prompt | llm` instead
llm_chain = prompt | llm | StrOutputParser()

result = llm_chain.invoke({"sentence": "the cat is on the table", "language": "spanish"})

# Affichage du résultat
print(result)