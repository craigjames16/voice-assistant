from dotenv import load_dotenv

load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.google.gmail.base import GmailToolSpec

tool_spec = GmailToolSpec()

llm = OpenAI(model="gpt-4o", temperature=0)

agent = ReActAgent.from_tools(tool_spec.to_tool_list(), llm=llm, verbose=True)

def chat(message: str):
    # Example function to send an email with the given message
    print(f"message: {message}")
    response = agent.chat(message)
    return response
