from typing import Optional
from pydantic import BaseModel
from pydantic_ai import RunContext
import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle
import base64
from email.mime.text import MIMEText

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
    
    if os.path.exists('agent/token.pickle'):
        with open('agent/token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'agent/credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open('agent/token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return build('gmail', 'v1', credentials=creds)

async def fetch_recent_emails(ctx: RunContext, max_results: int = 20) -> str:
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

async def create_draft_email(
    ctx: RunContext, 
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

async def send_draft(ctx: RunContext, draft_id: str) -> str:
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