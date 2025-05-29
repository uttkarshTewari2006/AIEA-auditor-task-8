import langchain
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import getpass
import os

load_dotenv()


if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

def file_to_string(file_path):
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
        return file_content
    except FileNotFoundError:
        return "Error: File not found"

# Example usage:
file_path = 'nl_prompt.txt'
file_string = file_to_string(file_path)
print(file_string)

model.invoke("Translate the following text to First Order Logic: " + file_string)

