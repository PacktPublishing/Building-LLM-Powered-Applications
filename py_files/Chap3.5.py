
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

system_message = """
You are a Python expert who produces Python code as per the user's request.
===>START EXAMPLE
---User Query---
Give me a function to print a string of text.
---User Output---
Below you can find the described function:
```def my_print(text):
     return print(text)
```
<===END EXAMPLE
"""
query = "generate a Python function to calculate the nth Fibonacci number"

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": system_message},
    {"role": "user", "content": query}]
)

#print(response)

print(response.choices[0].message.content)

