
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

system_message = """
You are an AI assistant specialized in solving riddles.
Given a riddle, solve it the best you can.
Provide a clear justification of your answer and the reasoning behind it.
Riddle:
"""

riddle = """
What has a face and two hands, but no arms or legs?
"""

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": system_message},
    {"role": "user", "content": riddle}]
)

#print(response)

print(response.choices[0].message.content)

