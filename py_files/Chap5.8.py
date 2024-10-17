import os

from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_openai import OpenAI

os.environ["OPENAI_API_KEY"]

memory = ConversationSummaryMemory(llm=OpenAI(temperature=0))
memory.save_context({"input": "hi, I'm looking for some ideas to write an essay in AI"}, {"output": "hello, what about writing on LLMs?"})
memory.load_memory_variables({})

print(memory.buffer)



