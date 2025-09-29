#!/usr/bin/env python3
"""
Google Gmail Backup Script
This script backs up Gmail emails using the Gmail API.
"""

import os
import json
import base64
import pickle
from datetime import datetime
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from email.mime.text import MIMEText
import argparse

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

class GmailBackup:
    def __init__(self, credentials_file='credentials.json', token_file='token.pickle'):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        
    def authenticate(self):
        """Authenticate and build Gmail service"""
        creds = None
        
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    print(f"Error: {self.credentials_file} not found!")
                    print("Please download your OAuth2 credentials from Google Cloud Console")
                    return False
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        self.service = build('gmail', 'v1', credentials=creds)
        return True
    
    def get_messages(self, query='', max_results=100):
        """Get list of messages matching query"""
        try:
            result = self.service.users().messages().list(
                userId='me', q=query, maxResults=max_results).execute()
            messages = result.get('messages', [])
            return messages
        except Exception as error:
            print(f'Error getting messages: {error}')
            return []
    
    def get_message_details(self, message_id):
        """Get detailed message content"""
        try:
            message = self.service.users().messages().get(
                userId='me', id=message_id, format='full').execute()
            return message
        except Exception as error:
            print(f'Error getting message {message_id}: {error}')
            return None
    
    def extract_message_data(self, message):
        """Extract relevant data from message"""
        payload = message.get('payload', {})
        headers = payload.get('headers', [])
        
        msg_data = {
            'id': message.get('id'),
            'thread_id': message.get('threadId'),
            'snippet': message.get('snippet'),
            'date': None,
            'from': None,
            'to': None,
            'subject': None,
            'body': None
        }
        
        for header in headers:
            name = header.get('name', '').lower()
            value = header.get('value', '')
            
            if name == 'date':
                msg_data['date'] = value
            elif name == 'from':
                msg_data['from'] = value
            elif name == 'to':
                msg_data['to'] = value
            elif name == 'subject':
                msg_data['subject'] = value
        
        msg_data['body'] = self.extract_body(payload)
        return msg_data
    
    def extract_body(self, payload):
        """Extract email body from payload"""
        body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body']['data']
                    body = base64.urlsafe_b64decode(data).decode('utf-8')
                    break
                elif part['mimeType'] == 'text/html' and not body:
                    data = part['body']['data']
                    body = base64.urlsafe_b64decode(data).decode('utf-8')
        else:
            if payload['mimeType'] == 'text/plain':
                data = payload['body']['data']
                body = base64.urlsafe_b64decode(data).decode('utf-8')
        
        return body
    
    def backup_emails(self, query='', max_results=100, output_dir='gmail_backup'):
        """Main backup function"""
        if not self.authenticate():
            return False
        
        print(f"Getting messages with query: '{query}'")
        messages = self.get_messages(query, max_results)
        
        if not messages:
            print("No messages found.")
            return True
        
        print(f"Found {len(messages)} messages")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        backup_data = []
        
        for i, message in enumerate(messages):
            print(f"Processing message {i+1}/{len(messages)}")
            
            msg_details = self.get_message_details(message['id'])
            if msg_details:
                msg_data = self.extract_message_data(msg_details)
                backup_data.append(msg_data)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f'gmail_backup_{timestamp}.json')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        
        print(f"Backup completed! Saved {len(backup_data)} messages to {output_file}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Backup Gmail emails')
    parser.add_argument('--query', default='', help='Gmail search query (e.g., "from:example@gmail.com")')
    parser.add_argument('--max', type=int, default=100, help='Maximum number of messages to backup')
    parser.add_argument('--output', default='gmail_backup', help='Output directory')
    parser.add_argument('--credentials', default='credentials.json', help='Path to credentials file')
    
    args = parser.parse_args()
    
    backup = GmailBackup(credentials_file=args.credentials)
    success = backup.backup_emails(
        query=args.query,
        max_results=args.max,
        output_dir=args.output
    )
    
    if success:
        print("Backup process completed successfully!")
    else:
        print("Backup process failed!")

if __name__ == '__main__':
    main()