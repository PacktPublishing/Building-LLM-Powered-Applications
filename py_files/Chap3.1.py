
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

system_message = """
You are an AI assistant that summarizes articles.
To complete this task, do the following subtasks:
Read the provided article context comprehensively and identify the main topic and key points
Generate a paragraph summary of the current article context that captures the essential information and conveys the main idea
Print each step of the process.
Article:
"""
article = """
Recurrent neural networks, long short-term memory, and gated recurrent neural networks
in particular, […]
"""

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": system_message},
    {"role": "user", "content": article}]
)

#print(response)

print(response.choices[0].message.content)

