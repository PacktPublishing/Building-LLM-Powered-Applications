
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint

repo_id = "tiiuae/falcon-7b-instruct" 

llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=1000, temperature=0.5,huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"])

print(llm.invoke("what was the first disney movie?"))
