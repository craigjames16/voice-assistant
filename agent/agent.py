from typing import Optional
from pydantic import BaseModel
from pydantic_ai import Agent
from . import gmail_tools

class UserQuery(BaseModel):
    """Structure for user queries and responses"""
    question: str

class QueryResponse(BaseModel):
    """Structure for agent responses"""
    answer: str
    confidence: float
    source: Optional[str] = None
    requires_followup: bool = False  # Add this field

# Initialize the agent with structured input/output
agent = Agent(
    'openai:gpt-4o',  # Using GPT-4 model
    deps_type=UserQuery,  # Input type
    result_type=QueryResponse,  # Output type
    system_prompt=(
        "You are a helpful AI assistant that provides clear, accurate answers. "
        "Always aim to be informative while maintaining high confidence in your responses. "
        "If you're not sure about something, indicate lower confidence. "
        "Try to answer in a conversational manner and not too long. "
        "When you need to ask a follow-up question, set requires_followup=true in your response. "
        "Otherwise, set requires_followup=false."
    )
)

def process_query_sync(question: str) -> QueryResponse:
    """Synchronous version of process_query"""
    query = UserQuery(question=question)
    
    result = agent.run_sync(
        question,
        deps=query
    )
    
    return result.data

# Register tools from gmail_tools
agent.tool(gmail_tools.fetch_recent_emails)
agent.tool(gmail_tools.create_draft_email)
agent.tool(gmail_tools.send_draft)

# result = process_query_sync("Create a new email draft to chisholm.craig@gmail.com telling them I'm going to be late")
# print(result)