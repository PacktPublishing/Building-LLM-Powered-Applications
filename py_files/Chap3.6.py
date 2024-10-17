
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

system_message = """
You are an AI marketing assistant. You help users to create taglines for new product names.
Given a product name, produce a tagline similar to the following examples:
Peak Pursuit - Conquer Heights with Comfort
Summit Steps - Your Partner for Every Ascent
Crag Conquerors - Step Up, Stand Tall
Product name:
"""
product_name = 'Elevation Embrace'

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": system_message},
    {"role": "user", "content": product_name}]
)

#print(response)

print(response.choices[0].message.content)

