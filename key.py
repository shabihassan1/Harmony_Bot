import os
from dotenv import load_dotenv

load_dotenv()

langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
