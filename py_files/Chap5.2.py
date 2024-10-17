
import os

from langchain_core.prompts import PromptTemplate

template = """Sentence: {sentence}
Translation in {language}:"""
prompt = PromptTemplate(template=template, input_variables=["sentence", "language"])

print(prompt.format(sentence = "the cat is on the table", language = "spanish"))