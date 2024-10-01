
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.chains.router import MultiPromptChain
from langchain.chains import ConversationChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

os.environ["OPENAI_API_KEY"]

llm = OpenAI()

itinerary_template = """You are a vacation itinerary assistant. \
You help customers finding the best destinations and itinerary. \
You help customer screating an optimized itinerary based on their preferences.

Here is a question:
{input}"""

restaurant_template = """You are a restaurant booking assitant. \
You check with customers number of guests and food preferences. \
You pay attention whether there are special conditions to take into account.

Here is a question:
{input}"""

prompt_infos = [
    {
        "name": "itinerary",
        "description": "Good for creating itinerary",
        "prompt_template": itinerary_template,
    },
    {
        "name": "restaurant",
        "description": "Good for help customers booking at restaurant",
        "prompt_template": restaurant_template,
    },
]

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain=LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

default_chain = ConversationChain(llm=llm, output_key="text")

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)

print(chain.invoke("I'm planning a trip from Milan to Venice by car. What can I visit in between?"))


print(chain.invoke("I want to book a table for tonight"))