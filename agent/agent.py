from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle
import base64
from email.mime.text import MIMEText

class UserQuery(BaseModel):
    """Structure for user queries and responses"""
    question: str

class QueryResponse(BaseModel):
    """Structure for agent responses"""
    answer: str
    confidence: float
    source: Optional[str] = None

# Initialize the agent with structured input/output
agent = Agent(
    'openai:gpt-4o',  # Using GPT-4 model
    deps_type=UserQuery,  # Input type
    result_type=QueryResponse,  # Output type
    system_prompt=(
        "You are a helpful AI assistant that provides clear, accurate answers. "
        "Always aim to be informative while maintaining high confidence in your responses. "
        "If you're not sure about something, indicate lower confidence."
        "Try to answer in a conversational manner and not too long"
    )
)

async def process_query(question: str) -> QueryResponse:
    """Process a user query and return a structured response"""
    print(f"Processing query: {question}")
    query = UserQuery(question=question)
    
    result = await agent.run(
        question,
        deps=query
    )
    return result.data

def process_query_sync(question: str) -> QueryResponse:
    """Synchronous version of process_query"""
    query = UserQuery(question=question)
    
    result = agent.run_sync(
        question,
        deps=query
    )
    
    return result.data

class EmailMessage(BaseModel):
    """Structure for email messages"""
    subject: str
    body: str
    to: str
    from_email: Optional[str] = None
    draft_id: Optional[str] = None

def get_gmail_service():
    """Helper function to authenticate and create Gmail service"""
    SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
    creds = None
    
    # Load credentials from token.pickle
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If credentials are invalid or don't exist, get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return build('gmail', 'v1', credentials=creds)

@agent.tool
async def fetch_recent_emails(ctx: RunContext[UserQuery], max_results: int = 20) -> str:
    """Fetch recent emails from Gmail"""
    service = get_gmail_service()
    results = service.users().messages().list(
        userId='me', maxResults=max_results, labelIds=['INBOX']
    ).execute()
    
    messages = []
    for msg in results.get('messages', []):
        message = service.users().messages().get(
            userId='me', id=msg['id']
        ).execute()
        
        headers = message['payload']['headers']
        subject = next(h['value'] for h in headers if h['name'] == 'Subject')
        sender = next(h['value'] for h in headers if h['name'] == 'From')
        
        messages.append(f"From: {sender}\nSubject: {subject}\n---")
    
    return "\n\n".join(messages)

@agent.tool
async def create_draft_email(
    ctx: RunContext[UserQuery], 
    subject: str,
    body: str,
    to: str
) -> str:
    """Create a draft email"""
    service = get_gmail_service()
    
    message = EmailMessage(
        subject=subject,
        body=body,
        to=to
    )
    
    mime_message = MIMEText(message.body)
    mime_message['to'] = message.to
    mime_message['subject'] = message.subject
    
    encoded_message = base64.urlsafe_b64encode(mime_message.as_bytes()).decode()
    
    draft = service.users().drafts().create(
        userId='me',
        body={
            'message': {
                'raw': encoded_message
            }
        }
    ).execute()
    
    return f"Draft created with ID: {draft['id']}"

@agent.tool
async def send_draft(ctx: RunContext[UserQuery], draft_id: str) -> str:
    """Send an existing draft email"""
    service = get_gmail_service()
    
    try:
        service.users().drafts().send(
            userId='me',
            body={'id': draft_id}
        ).execute()
        return f"Draft {draft_id} sent successfully"
    except Exception as e:
        return f"Error sending draft: {str(e)}"
    

# Add a new tool to fetch chat history
@agent.tool
async def fetch_chat_history(ctx: RunContext[UserQuery], limit: int = 10) -> str:
    """Fetch recent chat history"""
    messages = ctx.messages[-limit:]  # Get last N messages from context
    
    formatted_history = []
    for msg in messages:
        role_display = "User" if msg.role == 'user' else "Assistant"
        confidence_str = ""
        if msg.role == 'assistant' and hasattr(msg, 'data') and hasattr(msg.data, 'confidence'):
            confidence_str = f" (confidence: {msg.data.confidence:.2f})"
        msg_type_str = f" [{msg.type}]" if msg.type else ""
        formatted_history.append(f"{role_display}{confidence_str}{msg_type_str}\n{msg.content}\n")
    
    return "\n".join(formatted_history)

# result = process_query_sync("My name is Craig")
# print(result)
# result2 = process_query_sync("What is my name?")
# print(result2)