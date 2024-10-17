
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

system_message = """
You are a sentiment analyzer. You classify conversations into three categories: positive, negative, or neutral.
Return only the sentiment, in lowercase and without punctuation.
Conversation:
Remember to return only the sentiment, in lowercase and without punctuation
"""

conversation = """
Customer: Hi, I need some help with my order.
AI agent: Hello, welcome to our online store. I'm an AI agent and I'm here to assist you.
Customer: I ordered a pair of shoes yesterday, but I haven't received a confirmation email yet. Can you check the status of my order?
[…]
"""

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    #{"role": "system", "content": system_message},
    {"role": "user", "content": system_message}]
)

#print(response)

print(response.choices[0].message.content)

