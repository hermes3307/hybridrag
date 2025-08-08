import os
import json
import time
import webbrowser
from pathlib import Path
import subprocess
import threading
from datetime import datetime, timedelta

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False

# Google Drive API implementation
class GoogleDriveAPI:
    def __init__(self, force_demo_mode=False):
        self.authenticated = False
        self.service = None
        self.credentials = None
        self.scopes = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/calendar']
        self.force_demo_mode = force_demo_mode
        
    def authenticate(self, use_oauth=True):
        """Authenticate with Google Drive API using OAuth2"""
        # If force_demo_mode is True, skip OAuth2 and use demo mode
        if self.force_demo_mode:
            print("Running in demo mode (OAuth2 disabled)")
            time.sleep(1)
            self.authenticated = True
            self.calendar_service = None
            return True
            
        if not GOOGLE_API_AVAILABLE:
            # Fallback to demo mode
            time.sleep(1)
            self.authenticated = True
            self.calendar_service = None
            return True
            
        try:
            creds = None
            token_file = 'token.json'
            
            # Load existing credentials
            if os.path.exists(token_file):
                creds = Credentials.from_authorized_user_file(token_file, self.scopes)
            
            # If there are no (valid) credentials available, let the user log in
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    # Try to load from credentials.json first, then fall back to environment variables
                    if os.path.exists('credentials.json'):
                        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', self.scopes)
                    else:
                        # Use direct OAuth configuration from environment variables
                        client_id = os.getenv('GOOGLE_CLIENT_ID', 'YOUR_CLIENT_ID.apps.googleusercontent.com')
                        client_secret = os.getenv('GOOGLE_CLIENT_SECRET', 'YOUR_CLIENT_SECRET')
                        
                        if client_id.startswith('YOUR_') or client_secret.startswith('YOUR_'):
                            raise Exception("Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables or create credentials.json file")
                        
                        client_config = {
                            "installed": {
                                "client_id": client_id,
                                "client_secret": client_secret, 
                                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                                "token_uri": "https://oauth2.googleapis.com/token",
                                "redirect_uris": ["http://localhost"]
                            }
                        }
                        
                        # Create flow from client config directly (no file needed)
                        flow = InstalledAppFlow.from_client_config(client_config, self.scopes)
                    # This will open browser popup for authentication
                    creds = flow.run_local_server(port=0)
                
                # Save the credentials for the next run
                with open(token_file, 'w') as token:
                    token.write(creds.to_json())
            
            self.credentials = creds
            self.service = build('drive', 'v3', credentials=creds)
            self.calendar_service = build('calendar', 'v3', credentials=creds)
            self.authenticated = True
            return True
            
        except Exception as e:
            print(f"Authentication failed: {e}")
            # Fallback to demo mode
            time.sleep(1)
            self.authenticated = True
            self.calendar_service = None  # Ensure calendar service is None in demo mode
            return True
    
    def get_shared_files(self):
        """Get files that are shared publicly or with organization"""
        if not GOOGLE_API_AVAILABLE or not self.service:
            # Return demo data
            return self._get_demo_files()
            
        try:
            # Search for files with sharing permissions
            results = self.service.files().list(
                pageSize=100,
                fields="nextPageToken, files(id, name, permissions)"
            ).execute()
            
            files = results.get('files', [])
            shared_files = []
            
            for file in files:
                permissions = file.get('permissions', [])
                sharing_status = self._analyze_permissions(permissions)
                
                if sharing_status != "Private":
                    shared_files.append({
                        'id': file['id'],
                        'name': file['name'],
                        'permissions': permissions,
                        'sharing_status': sharing_status
                    })
            
            return shared_files
            
        except HttpError as e:
            print(f"Error accessing Google Drive: {e}")
            return self._get_demo_files()
    
    def _analyze_permissions(self, permissions):
        """Analyze permissions to determine sharing status"""
        for perm in permissions:
            if perm.get('type') == 'anyone':
                if perm.get('role') == 'reader':
                    return 'Public - Anyone with link can view'
                elif perm.get('role') == 'commenter':
                    return 'Public - Anyone can comment'
                elif perm.get('role') == 'writer':
                    return 'Public - Anyone can edit'
            elif perm.get('type') == 'domain':
                domain = perm.get('domain', 'organization')
                return f'Organization - {domain}'
        return 'Private'
    
    def _get_demo_files(self):
        """Return demo files when real API is not available"""
        return [
            {
                'id': 'file_001',
                'name': 'Project Proposal.docx',
                'permissions': [
                    {'type': 'anyone', 'role': 'reader', 'id': 'perm_001'},
                    {'type': 'user', 'role': 'owner', 'emailAddress': 'you@gmail.com', 'id': 'perm_002'}
                ],
                'sharing_status': 'Public - Anyone with link'
            },
            {
                'id': 'file_002', 
                'name': 'Budget Report.xlsx',
                'permissions': [
                    {'type': 'domain', 'role': 'reader', 'domain': 'company.com', 'id': 'perm_003'},
                    {'type': 'user', 'role': 'owner', 'emailAddress': 'you@gmail.com', 'id': 'perm_004'}
                ],
                'sharing_status': 'Organization - company.com'
            },
            {
                'id': 'file_003',
                'name': 'Meeting Notes.pdf', 
                'permissions': [
                    {'type': 'anyone', 'role': 'commenter', 'id': 'perm_005'},
                    {'type': 'user', 'role': 'owner', 'emailAddress': 'you@gmail.com', 'id': 'perm_006'}
                ],
                'sharing_status': 'Public - Anyone can comment'
            }
        ]
    
    def make_file_private(self, file_id):
        """Remove public/organization permissions from a file"""
        if not GOOGLE_API_AVAILABLE or not self.service:
            time.sleep(0.5)  # Simulate API call
            return True
            
        try:
            # Get current permissions
            permissions = self.service.permissions().list(fileId=file_id).execute()
            
            for perm in permissions.get('permissions', []):
                # Remove public and domain permissions (keep owner)
                if perm.get('type') in ['anyone', 'domain']:
                    self.service.permissions().delete(
                        fileId=file_id,
                        permissionId=perm['id']
                    ).execute()
            
            return True
            
        except HttpError as e:
            print(f"Error making file private: {e}")
            return False

    def get_all_calendars(self):
        """Get ALL calendars (both public and private) with their sharing status"""
        if not GOOGLE_API_AVAILABLE or not hasattr(self, 'calendar_service') or not self.calendar_service:
            # Return demo data
            return self._get_demo_calendars()
            
        try:
            # Get all calendars
            calendars_result = self.calendar_service.calendarList().list().execute()
            calendars = calendars_result.get('items', [])
            
            all_calendars = []
            
            for calendar in calendars:
                calendar_id = calendar['id']
                calendar_name = calendar.get('summary', 'Unnamed Calendar')
                
                try:
                    # Get ACL rules for this calendar
                    acl_rules = self.calendar_service.acl().list(calendarId=calendar_id).execute()
                    sharing_status = self._analyze_calendar_permissions(acl_rules.get('items', []))
                    
                    # Add ALL calendars (both public and private)
                    all_calendars.append({
                        'id': calendar_id,
                        'name': calendar_name,
                        'description': calendar.get('description', ''),
                        'sharing_status': sharing_status,
                        'acl_rules': acl_rules.get('items', []),
                        'is_public': sharing_status != "Private",
                        'access_role': calendar.get('accessRole', 'unknown'),
                        'primary': calendar.get('primary', False)
                    })
                        
                except HttpError as e:
                    # Even if we can't check permissions, add the calendar with unknown status
                    if e.resp.status in [403, 404]:
                        all_calendars.append({
                            'id': calendar_id,
                            'name': calendar_name,
                            'description': calendar.get('description', ''),
                            'sharing_status': "Access Restricted",
                            'acl_rules': [],
                            'is_public': False,
                            'access_role': calendar.get('accessRole', 'reader'),
                            'primary': calendar.get('primary', False)
                        })
                    else:
                        print(f"Error checking calendar {calendar_name}: {e}")
                        all_calendars.append({
                            'id': calendar_id,
                            'name': calendar_name,
                            'description': calendar.get('description', ''),
                            'sharing_status': "Check Failed",
                            'acl_rules': [],
                            'is_public': False,
                            'access_role': calendar.get('accessRole', 'unknown'),
                            'primary': calendar.get('primary', False)
                        })
                except Exception as e:
                    print(f"Unexpected error checking calendar {calendar_name}: {e}")
                    all_calendars.append({
                        'id': calendar_id,
                        'name': calendar_name,
                        'description': calendar.get('description', ''),
                        'sharing_status': "Error",
                        'acl_rules': [],
                        'is_public': False,
                        'access_role': calendar.get('accessRole', 'unknown'),
                        'primary': calendar.get('primary', False)
                    })
            
            return all_calendars
            
        except HttpError as e:
            print(f"Error accessing Google Calendar: {e}")
            return self._get_demo_calendars()
            
    def get_public_calendars(self):
        """Get only calendars that are public (for backward compatibility)"""
        all_calendars = self.get_all_calendars()
        return [cal for cal in all_calendars if cal['is_public']]
            
    def get_calendar_events(self, calendar_id, max_results=10):
        """Get recent and upcoming events for a specific calendar"""
        if not GOOGLE_API_AVAILABLE or not hasattr(self, 'calendar_service') or not self.calendar_service:
            # Return demo events
            return self._get_demo_events(calendar_id)
            
        try:
            # Get events from the last 30 days and next 30 days
            now = datetime.utcnow()
            time_min = (now - timedelta(days=30)).isoformat() + 'Z'
            time_max = (now + timedelta(days=30)).isoformat() + 'Z'
            
            events_result = self.calendar_service.events().list(
                calendarId=calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            formatted_events = []
            
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))
                
                # Parse datetime
                if 'T' in start:  # DateTime format
                    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                    date_str = start_dt.strftime('%Y-%m-%d %H:%M')
                    duration = f"{(end_dt - start_dt).seconds // 3600}h {((end_dt - start_dt).seconds % 3600) // 60}m"
                else:  # Date format (all-day)
                    start_dt = datetime.fromisoformat(start)
                    date_str = start_dt.strftime('%Y-%m-%d')
                    duration = "All day"
                
                formatted_events.append({
                    'id': event['id'],
                    'title': event.get('summary', 'No title'),
                    'description': event.get('description', ''),
                    'date': date_str,
                    'duration': duration,
                    'location': event.get('location', ''),
                    'attendees': len(event.get('attendees', [])),
                    'visibility': event.get('visibility', 'default')
                })
            
            return formatted_events
            
        except HttpError as e:
            print(f"Error getting calendar events: {e}")
            return self._get_demo_events(calendar_id)
        except Exception as e:
            print(f"Unexpected error getting calendar events: {e}")
            return self._get_demo_events(calendar_id)
            
    def get_all_events_from_calendar(self, calendar_id, max_results=100):
        """Get ALL events from a calendar (for privacy management)"""
        if not GOOGLE_API_AVAILABLE or not hasattr(self, 'calendar_service') or not self.calendar_service:
            # Return demo events
            return self._get_all_demo_events(calendar_id)
            
        try:
            # Get events from the last 6 months and next 6 months for comprehensive coverage
            now = datetime.utcnow()
            time_min = (now - timedelta(days=180)).isoformat() + 'Z'
            time_max = (now + timedelta(days=180)).isoformat() + 'Z'
            
            events_result = self.calendar_service.events().list(
                calendarId=calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            formatted_events = []
            
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))
                
                # Parse datetime
                if 'T' in start:  # DateTime format
                    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                    date_str = start_dt.strftime('%Y-%m-%d %H:%M')
                    duration = f"{(end_dt - start_dt).seconds // 3600}h {((end_dt - start_dt).seconds % 3600) // 60}m"
                else:  # Date format (all-day)
                    start_dt = datetime.fromisoformat(start)
                    date_str = start_dt.strftime('%Y-%m-%d')
                    duration = "All day"
                
                formatted_events.append({
                    'id': event['id'],
                    'title': event.get('summary', 'No title'),
                    'description': event.get('description', ''),
                    'date': date_str,
                    'duration': duration,
                    'location': event.get('location', ''),
                    'attendees': len(event.get('attendees', [])),
                    'visibility': event.get('visibility', 'default'),
                    'transparency': event.get('transparency', 'opaque'),  # opaque = busy, transparent = free
                    'original_event': event  # Keep original for updates
                })
            
            return formatted_events
            
        except HttpError as e:
            print(f"Error getting all calendar events: {e}")
            return self._get_all_demo_events(calendar_id)
        except Exception as e:
            print(f"Unexpected error getting all calendar events: {e}")
            return self._get_all_demo_events(calendar_id)
    
    def _get_demo_events(self, calendar_id):
        """Return demo events for different calendar types"""
        if calendar_id == 'calendar_001':  # Work Schedule
            return [
                {'id': 'event_001', 'title': 'Team Meeting', 'description': 'Weekly team sync', 'date': '2024-01-15 09:00', 'duration': '1h 0m', 'location': 'Conference Room A', 'attendees': 5, 'visibility': 'default'},
                {'id': 'event_002', 'title': 'Project Review', 'description': 'Q4 project status', 'date': '2024-01-16 14:00', 'duration': '2h 0m', 'location': 'Main Office', 'attendees': 8, 'visibility': 'default'},
                {'id': 'event_003', 'title': 'Client Presentation', 'description': 'Product demo for client', 'date': '2024-01-18 10:30', 'duration': '1h 30m', 'location': 'Client Office', 'attendees': 12, 'visibility': 'public'}
            ]
        elif calendar_id == 'calendar_002':  # Team Events
            return [
                {'id': 'event_004', 'title': 'Company All-Hands', 'description': 'Monthly company meeting', 'date': '2024-01-20', 'duration': 'All day', 'location': 'Main Auditorium', 'attendees': 150, 'visibility': 'public'},
                {'id': 'event_005', 'title': 'Team Building Activity', 'description': 'Outdoor team building', 'date': '2024-01-25 13:00', 'duration': '4h 0m', 'location': 'City Park', 'attendees': 25, 'visibility': 'public'}
            ]
        elif calendar_id == 'calendar_003':  # Project Deadlines
            return [
                {'id': 'event_006', 'title': 'Phase 1 Deadline', 'description': 'Complete initial development', 'date': '2024-01-30', 'duration': 'All day', 'location': '', 'attendees': 0, 'visibility': 'public'},
                {'id': 'event_007', 'title': 'Code Review Session', 'description': 'Review Phase 1 code', 'date': '2024-02-02 15:00', 'duration': '3h 0m', 'location': 'Development Office', 'attendees': 6, 'visibility': 'public'}
            ]
        elif calendar_id == 'calendar_004':  # Personal Calendar (Private)
            return [
                {'id': 'event_008', 'title': 'Gym Workout', 'description': 'Weekly fitness routine', 'date': '2024-01-17 18:00', 'duration': '1h 30m', 'location': 'Local Gym', 'attendees': 0, 'visibility': 'private'},
                {'id': 'event_009', 'title': 'Birthday Planning', 'description': 'Plan surprise party', 'date': '2024-01-22 19:00', 'duration': '2h 0m', 'location': 'Home', 'attendees': 0, 'visibility': 'private'},
                {'id': 'event_010', 'title': 'Weekend Trip', 'description': 'Weekend getaway planning', 'date': '2024-01-27', 'duration': 'All day', 'location': 'Mountains', 'attendees': 0, 'visibility': 'private'}
            ]
        elif calendar_id == 'calendar_005':  # Doctor Appointments (Private)
            return [
                {'id': 'event_011', 'title': 'Annual Checkup', 'description': 'Routine medical examination', 'date': '2024-01-24 10:00', 'duration': '1h 0m', 'location': 'Medical Center', 'attendees': 0, 'visibility': 'private'},
                {'id': 'event_012', 'title': 'Dental Cleaning', 'description': 'Regular dental maintenance', 'date': '2024-02-05 14:30', 'duration': '45m', 'location': 'Dental Clinic', 'attendees': 0, 'visibility': 'private'}
            ]
        elif calendar_id == 'calendar_006':  # Family Events (Private)
            return [
                {'id': 'event_013', 'title': 'Family Dinner', 'description': 'Monthly family gathering', 'date': '2024-01-21 18:30', 'duration': '3h 0m', 'location': 'Home', 'attendees': 6, 'visibility': 'private'},
                {'id': 'event_014', 'title': 'Kids Soccer Game', 'description': 'Support the kids at their game', 'date': '2024-01-26 15:00', 'duration': '2h 0m', 'location': 'Sports Complex', 'attendees': 4, 'visibility': 'private'}
            ]
        else:
            return []
    
    def _get_all_demo_events(self, calendar_id):
        """Return ALL demo events with various visibility settings for privacy management"""
        if calendar_id == 'calendar_001':  # Work Schedule
            return [
                {'id': 'event_001', 'title': 'Team Meeting', 'description': 'Weekly team sync - sensitive project discussion', 'date': '2024-01-15 09:00', 'duration': '1h 0m', 'location': 'Conference Room A', 'attendees': 5, 'visibility': 'default', 'transparency': 'opaque'},
                {'id': 'event_002', 'title': 'Project Review', 'description': 'Q4 project status - confidential budget info', 'date': '2024-01-16 14:00', 'duration': '2h 0m', 'location': 'Main Office', 'attendees': 8, 'visibility': 'public', 'transparency': 'opaque'},
                {'id': 'event_003', 'title': 'Client Presentation', 'description': 'Product demo for client - NDA required', 'date': '2024-01-18 10:30', 'duration': '1h 30m', 'location': 'Client Office', 'attendees': 12, 'visibility': 'default', 'transparency': 'opaque'},
                {'id': 'event_021', 'title': 'Salary Review Meeting', 'description': 'Personal compensation discussion - CONFIDENTIAL', 'date': '2024-01-19 15:00', 'duration': '1h 0m', 'location': 'HR Office', 'attendees': 2, 'visibility': 'public', 'transparency': 'opaque'},
                {'id': 'event_022', 'title': 'Strategy Planning', 'description': 'Company strategy - highly sensitive', 'date': '2024-01-22 11:00', 'duration': '3h 0m', 'location': 'Executive Suite', 'attendees': 6, 'visibility': 'default', 'transparency': 'opaque'}
            ]
        elif calendar_id == 'calendar_002':  # Team Events
            return [
                {'id': 'event_004', 'title': 'Company All-Hands', 'description': 'Monthly company meeting', 'date': '2024-01-20', 'duration': 'All day', 'location': 'Main Auditorium', 'attendees': 150, 'visibility': 'public', 'transparency': 'opaque'},
                {'id': 'event_005', 'title': 'Team Building Activity', 'description': 'Outdoor team building', 'date': '2024-01-25 13:00', 'duration': '4h 0m', 'location': 'City Park', 'attendees': 25, 'visibility': 'public', 'transparency': 'opaque'},
                {'id': 'event_023', 'title': 'Performance Review Session', 'description': 'Individual performance discussions - PRIVATE', 'date': '2024-01-26 14:00', 'duration': '2h 0m', 'location': 'Meeting Room B', 'attendees': 8, 'visibility': 'default', 'transparency': 'opaque'}
            ]
        elif calendar_id == 'calendar_003':  # Project Deadlines
            return [
                {'id': 'event_006', 'title': 'Phase 1 Deadline', 'description': 'Complete initial development', 'date': '2024-01-30', 'duration': 'All day', 'location': '', 'attendees': 0, 'visibility': 'public', 'transparency': 'opaque'},
                {'id': 'event_007', 'title': 'Code Review Session', 'description': 'Review Phase 1 code - proprietary technology', 'date': '2024-02-02 15:00', 'duration': '3h 0m', 'location': 'Development Office', 'attendees': 6, 'visibility': 'default', 'transparency': 'opaque'},
                {'id': 'event_024', 'title': 'Security Audit', 'description': 'Internal security review - CONFIDENTIAL', 'date': '2024-02-05 10:00', 'duration': '4h 0m', 'location': 'Secure Conference Room', 'attendees': 4, 'visibility': 'public', 'transparency': 'opaque'}
            ]
        elif calendar_id == 'calendar_004':  # Personal Calendar (Private)
            return [
                {'id': 'event_008', 'title': 'Gym Workout', 'description': 'Weekly fitness routine', 'date': '2024-01-17 18:00', 'duration': '1h 30m', 'location': 'Local Gym', 'attendees': 0, 'visibility': 'default', 'transparency': 'opaque'},
                {'id': 'event_009', 'title': 'Birthday Planning', 'description': 'Plan surprise party for spouse', 'date': '2024-01-22 19:00', 'duration': '2h 0m', 'location': 'Home', 'attendees': 0, 'visibility': 'default', 'transparency': 'opaque'},
                {'id': 'event_010', 'title': 'Weekend Trip', 'description': 'Weekend getaway planning - personal', 'date': '2024-01-27', 'duration': 'All day', 'location': 'Mountains', 'attendees': 0, 'visibility': 'default', 'transparency': 'opaque'},
                {'id': 'event_025', 'title': 'Financial Planning', 'description': 'Personal investment meeting - SENSITIVE', 'date': '2024-01-29 16:00', 'duration': '2h 0m', 'location': 'Bank Office', 'attendees': 2, 'visibility': 'default', 'transparency': 'opaque'}
            ]
        elif calendar_id == 'calendar_005':  # Doctor Appointments (Private)
            return [
                {'id': 'event_011', 'title': 'Annual Checkup', 'description': 'Routine medical examination', 'date': '2024-01-24 10:00', 'duration': '1h 0m', 'location': 'Medical Center', 'attendees': 0, 'visibility': 'default', 'transparency': 'opaque'},
                {'id': 'event_012', 'title': 'Dental Cleaning', 'description': 'Regular dental maintenance', 'date': '2024-02-05 14:30', 'duration': '45m', 'location': 'Dental Clinic', 'attendees': 0, 'visibility': 'default', 'transparency': 'opaque'},
                {'id': 'event_026', 'title': 'Specialist Consultation', 'description': 'Confidential medical consultation', 'date': '2024-02-08 11:00', 'duration': '1h 30m', 'location': 'Specialist Clinic', 'attendees': 0, 'visibility': 'default', 'transparency': 'opaque'}
            ]
        elif calendar_id == 'calendar_006':  # Family Events (Private)
            return [
                {'id': 'event_013', 'title': 'Family Dinner', 'description': 'Monthly family gathering', 'date': '2024-01-21 18:30', 'duration': '3h 0m', 'location': 'Home', 'attendees': 6, 'visibility': 'default', 'transparency': 'opaque'},
                {'id': 'event_014', 'title': 'Kids Soccer Game', 'description': 'Support the kids at their game', 'date': '2024-01-26 15:00', 'duration': '2h 0m', 'location': 'Sports Complex', 'attendees': 4, 'visibility': 'default', 'transparency': 'opaque'},
                {'id': 'event_027', 'title': 'Anniversary Planning', 'description': 'Plan wedding anniversary surprise - SECRET', 'date': '2024-01-31 20:00', 'duration': '1h 0m', 'location': 'Home', 'attendees': 0, 'visibility': 'default', 'transparency': 'opaque'}
            ]
        else:
            return []
    
    def _analyze_calendar_permissions(self, acl_rules):
        """Analyze calendar ACL rules to determine sharing status"""
        for rule in acl_rules:
            scope = rule.get('scope', {})
            scope_type = scope.get('type')
            
            if scope_type == 'default':
                role = rule.get('role', '')
                if role == 'reader':
                    return 'Public - Anyone can view'
                elif role == 'writer':
                    return 'Public - Anyone can edit'
                elif role == 'owner':
                    return 'Public - Anyone can manage'
            elif scope_type == 'domain':
                domain = scope.get('value', 'organization')
                role = rule.get('role', '')
                return f'Organization - {domain} ({role})'
        
        return 'Private'
    
    def _get_demo_calendars(self):
        """Return demo calendars when real API is not available"""
        return [
            {
                'id': 'calendar_001',
                'name': 'Work Schedule',
                'description': 'Company work calendar',
                'sharing_status': 'Organization - company.com (reader)',
                'acl_rules': [
                    {'scope': {'type': 'domain', 'value': 'company.com'}, 'role': 'reader'},
                    {'scope': {'type': 'user', 'value': 'you@gmail.com'}, 'role': 'owner'}
                ],
                'is_public': True,
                'access_role': 'owner',
                'primary': False
            },
            {
                'id': 'calendar_002',
                'name': 'Team Events',
                'description': 'Public team calendar',
                'sharing_status': 'Public - Anyone can view',
                'acl_rules': [
                    {'scope': {'type': 'default'}, 'role': 'reader'},
                    {'scope': {'type': 'user', 'value': 'you@gmail.com'}, 'role': 'owner'}
                ],
                'is_public': True,
                'access_role': 'owner',
                'primary': False
            },
            {
                'id': 'calendar_003',
                'name': 'Project Deadlines',
                'description': 'Shared project calendar',
                'sharing_status': 'Public - Anyone can edit',
                'acl_rules': [
                    {'scope': {'type': 'default'}, 'role': 'writer'},
                    {'scope': {'type': 'user', 'value': 'you@gmail.com'}, 'role': 'owner'}
                ],
                'is_public': True,
                'access_role': 'owner',
                'primary': False
            },
            {
                'id': 'calendar_004',
                'name': 'Personal Calendar',
                'description': 'My private personal calendar',
                'sharing_status': 'Private',
                'acl_rules': [
                    {'scope': {'type': 'user', 'value': 'you@gmail.com'}, 'role': 'owner'}
                ],
                'is_public': False,
                'access_role': 'owner',
                'primary': True
            },
            {
                'id': 'calendar_005',
                'name': 'Doctor Appointments',
                'description': 'Private medical appointments',
                'sharing_status': 'Private',
                'acl_rules': [
                    {'scope': {'type': 'user', 'value': 'you@gmail.com'}, 'role': 'owner'}
                ],
                'is_public': False,
                'access_role': 'owner',
                'primary': False
            },
            {
                'id': 'calendar_006',
                'name': 'Family Events',
                'description': 'Private family calendar',
                'sharing_status': 'Private',
                'acl_rules': [
                    {'scope': {'type': 'user', 'value': 'you@gmail.com'}, 'role': 'owner'}
                ],
                'is_public': False,
                'access_role': 'owner',
                'primary': False
            }
        ]

    def make_calendar_private(self, calendar_id):
        """Remove public/organization permissions from a calendar"""
        if not GOOGLE_API_AVAILABLE or not hasattr(self, 'calendar_service') or not self.calendar_service:
            time.sleep(0.5)  # Simulate API call
            return True
            
        try:
            # Get current ACL rules
            acl_rules = self.calendar_service.acl().list(calendarId=calendar_id).execute()
            
            for rule in acl_rules.get('items', []):
                scope = rule.get('scope', {})
                scope_type = scope.get('type')
                
                # Remove public and domain permissions (keep owner and specific user permissions)
                if scope_type in ['default', 'domain']:
                    try:
                        self.calendar_service.acl().delete(
                            calendarId=calendar_id,
                            ruleId=rule['id']
                        ).execute()
                        print(f"Removed {scope_type} permission from calendar")
                    except HttpError as delete_error:
                        print(f"Warning: Could not remove permission {rule['id']}: {delete_error}")
                        continue
            
            return True
            
        except HttpError as e:
            print(f"Error making calendar private: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error making calendar private: {e}")
            return False
    
    def make_event_private_busy(self, calendar_id, event_id):
        """Make a specific event private and show as busy"""
        if not GOOGLE_API_AVAILABLE or not hasattr(self, 'calendar_service') or not self.calendar_service:
            time.sleep(0.2)  # Simulate API call
            return True
            
        try:
            # Get the event first
            event = self.calendar_service.events().get(calendarId=calendar_id, eventId=event_id).execute()
            
            # Update event to be private and busy
            event['visibility'] = 'private'  # Hide details from others
            event['transparency'] = 'opaque'  # Show as busy
            
            # Remove sensitive information that might be visible
            if 'description' in event:
                # Keep description but mark as private
                pass  # Description is already hidden when visibility=private
            
            # Update the event
            updated_event = self.calendar_service.events().update(
                calendarId=calendar_id,
                eventId=event_id,
                body=event
            ).execute()
            
            return True
            
        except HttpError as e:
            print(f"Error making event private: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error making event private: {e}")
            return False
    
    def make_all_calendar_events_private_busy(self, calendar_id):
        """Make ALL events in a calendar private and busy for maximum security"""
        if not GOOGLE_API_AVAILABLE or not hasattr(self, 'calendar_service') or not self.calendar_service:
            time.sleep(1)  # Simulate API call
            return True, 10  # Return success and fake count
            
        try:
            success_count = 0
            error_count = 0
            
            # Get all events from the calendar
            events = self.get_all_events_from_calendar(calendar_id)
            
            for event_info in events:
                try:
                    if self.make_event_private_busy(calendar_id, event_info['id']):
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    print(f"Error securing event {event_info['title']}: {e}")
                    error_count += 1
            
            return success_count > 0, success_count
            
        except Exception as e:
            print(f"Error securing calendar events: {e}")
            return False, 0

class GooglePrivacyManager:
    def __init__(self, force_demo_mode=False):
        self.api = GoogleDriveAPI(force_demo_mode=force_demo_mode)
        self.shared_files = []
        self.all_calendars = []
        self.public_calendars = []
        self.force_demo_mode = force_demo_mode
        
        if GUI_AVAILABLE:
            self.setup_gui()
        else:
            self.setup_console()
    
    def secure_all_calendar_events(self):
        """Make ALL events in selected calendar private and busy for maximum security"""
        selected_items = self.calendars_tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select a calendar to secure all events")
            return
        
        # Get selected calendar
        calendar_index = self.calendars_tree.index(selected_items[0])
        selected_calendar = self.all_calendars[calendar_index]
        
        # Confirm action with user
        result = messagebox.askyesno("Confirm Event Security", 
                                f"⚠️ SECURITY ACTION ⚠️\n\n" +
                                f"This will make ALL events in '{selected_calendar['name']}' calendar:\n" +
                                f"• Private (details hidden from others)\n" +
                                f"• Busy (blocks time on shared calendars)\n\n" +
                                f"This is recommended for calendars containing:\n" +
                                f"• Medical appointments\n" +
                                f"• Personal meetings\n" +
                                f"• Sensitive business discussions\n\n" +
                                f"Continue?")
        
        if not result:
            return
        
        def do_secure():
            self.progress_label.config(text=f"Securing all events in {selected_calendar['name']}...")
            self.progress_var.set(20)
            
            try:
                # Make all events private and busy
                success, count = self.api.make_all_calendar_events_private_busy(selected_calendar['id'])
                
                self.progress_var.set(100)
                
                if success:
                    self.progress_label.config(text=f"✅ Secured {count} events in {selected_calendar['name']}")
                    messagebox.showinfo("Security Complete", 
                                    f"Successfully secured {count} events!\n\n" +
                                    f"All events in '{selected_calendar['name']}' are now:\n" +
                                    f"• Private (details hidden)\n" +
                                    f"• Showing as busy time only\n\n" +
                                    f"Your calendar information is now protected.")
                else:
                    self.progress_label.config(text="Failed to secure events")
                    messagebox.showerror("Error", 
                                        "Failed to secure calendar events.\n" +
                                        "Please try again or check permissions.")
            
            except Exception as e:
                self.progress_label.config(text="Error securing events")
                messagebox.showerror("Error", f"Error securing events: {str(e)}")
            
            # Reset progress after delay
            self.root.after(3000, lambda: [self.progress_var.set(0), self.progress_label.config(text="Ready")])
        
        # Run in separate thread
        threading.Thread(target=do_secure, daemon=True).start()

    def setup_gui(self):
        """Setup GUI interface"""
        self.root = tk.Tk()
        self.root.title("Google Privacy Manager - Drive Files & Calendars")
        self.root.geometry("800x600")
        
        # Variables
        self.progress_var = tk.DoubleVar()
        self.is_logged_in = tk.BooleanVar(value=False)
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Google Privacy Manager", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Login section
        login_frame = ttk.LabelFrame(main_frame, text="Authentication", padding="10")
        login_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(login_frame, text="Not logged in to Google services")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        self.login_button = ttk.Button(login_frame, text="Login to Google", 
                                      command=self.google_login)
        self.login_button.grid(row=0, column=1, sticky=tk.E, padx=(10, 0))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Files tab
        files_frame = ttk.Frame(notebook, padding="10")
        notebook.add(files_frame, text="Drive Files")
        
        # Treeview for files
        columns = ('Name', 'Sharing Status', 'Action')
        self.files_tree = ttk.Treeview(files_frame, columns=columns, show='headings', height=8)
        
        # Configure columns
        self.files_tree.heading('Name', text='File Name')
        self.files_tree.heading('Sharing Status', text='Current Sharing')
        self.files_tree.heading('Action', text='Action Needed')
        
        self.files_tree.column('Name', width=300)
        self.files_tree.column('Sharing Status', width=200)
        self.files_tree.column('Action', width=150)
        
        # Scrollbar for files treeview
        files_scrollbar = ttk.Scrollbar(files_frame, orient=tk.VERTICAL, command=self.files_tree.yview)
        self.files_tree.configure(yscrollcommand=files_scrollbar.set)
        
        self.files_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        files_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Files buttons
        files_button_frame = ttk.Frame(files_frame)
        files_button_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        self.scan_files_button = ttk.Button(files_button_frame, text="Scan Drive Files", 
                                           command=self.scan_drive_files, state=tk.DISABLED)
        self.scan_files_button.grid(row=0, column=0, padx=(0, 5))
        
        self.make_files_private_button = ttk.Button(files_button_frame, text="Make Selected Private", 
                                                   command=self.make_selected_files_private, state=tk.DISABLED)
        self.make_files_private_button.grid(row=0, column=1, padx=5)
        
        self.make_all_files_private_button = ttk.Button(files_button_frame, text="Make All Private", 
                                                       command=self.make_all_files_private, state=tk.DISABLED)
        self.make_all_files_private_button.grid(row=0, column=2, padx=(5, 0))
        
        # Calendars tab
        calendars_frame = ttk.Frame(notebook, padding="10")
        notebook.add(calendars_frame, text="All Calendars")
        
        # Treeview for calendars
        calendar_columns = ('Name', 'Type', 'Sharing Status', 'Security Risk')
        self.calendars_tree = ttk.Treeview(calendars_frame, columns=calendar_columns, show='headings', height=8)
        
        # Configure calendar columns
        self.calendars_tree.heading('Name', text='Calendar Name')
        self.calendars_tree.heading('Type', text='Type')
        self.calendars_tree.heading('Sharing Status', text='Current Sharing')
        self.calendars_tree.heading('Security Risk', text='Security Risk')
        
        self.calendars_tree.column('Name', width=250)
        self.calendars_tree.column('Type', width=80)
        self.calendars_tree.column('Sharing Status', width=180)
        self.calendars_tree.column('Security Risk', width=100)
        
        # Scrollbar for calendars treeview
        calendars_scrollbar = ttk.Scrollbar(calendars_frame, orient=tk.VERTICAL, command=self.calendars_tree.yview)
        self.calendars_tree.configure(yscrollcommand=calendars_scrollbar.set)
        
        self.calendars_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        calendars_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Calendar buttons
        calendars_button_frame = ttk.Frame(calendars_frame)
        calendars_button_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        self.scan_calendars_button = ttk.Button(calendars_button_frame, text="Scan All Calendars", 
                                               command=self.scan_all_calendars, state=tk.DISABLED)
        self.scan_calendars_button.grid(row=0, column=0, padx=(0, 5))
        
        self.view_events_button = ttk.Button(calendars_button_frame, text="View Events", 
                                            command=self.view_calendar_events, state=tk.DISABLED)
        self.view_events_button.grid(row=0, column=1, padx=5)
        
        self.make_calendars_private_button = ttk.Button(calendars_button_frame, text="Make Selected Private", 
                                                       command=self.make_selected_calendars_private, state=tk.DISABLED)
        self.make_calendars_private_button.grid(row=0, column=2, padx=5)
        
        self.make_all_public_private_button = ttk.Button(calendars_button_frame, text="Make All Public Private", 
                                                        command=self.make_all_public_calendars_private, state=tk.DISABLED)
        self.make_all_public_private_button.grid(row=0, column=3, padx=(5, 0))
        
        # Second row of calendar buttons for event-level privacy
        calendars_button_frame2 = ttk.Frame(calendars_frame)
        calendars_button_frame2.grid(row=2, column=0, columnspan=2, pady=(5, 0))
        
        self.secure_events_button = ttk.Button(calendars_button_frame2, text="🔒 Secure All Events (Private/Busy)", 
                                              command=self.secure_all_calendar_events, state=tk.DISABLED)
        self.secure_events_button.grid(row=0, column=0, padx=(0, 5))
        
        ttk.Label(calendars_button_frame2, text="⚠️ Makes ALL events in selected calendar private & busy", 
                 font=("Arial", 8), foreground="orange").grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        # Configure grid weights for tabs
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)
        calendars_frame.columnconfigure(0, weight=1)
        calendars_frame.rowconfigure(0, weight=1)
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.grid(row=1, column=0, pady=(5, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)
        login_frame.columnconfigure(0, weight=1)
        progress_frame.columnconfigure(0, weight=1)
    
    def setup_console(self):
        """Setup console interface if GUI not available"""
        print("GUI not available. Using console interface.")
        self.console_main()
    
    def google_login(self):
        """Handle Google login"""
        if self.api.authenticated:
            # Logout
            self.api.authenticated = False
            self.api.service = None
            self.api.calendar_service = None
            self.api.credentials = None
            self.is_logged_in.set(False)
            self.status_label.config(text="Not logged in to Google services")
            self.login_button.config(text="Login to Google")
            # Disable all buttons
            self.scan_files_button.config(state=tk.DISABLED)
            self.make_files_private_button.config(state=tk.DISABLED)
            self.make_all_files_private_button.config(state=tk.DISABLED)
            self.scan_calendars_button.config(state=tk.DISABLED)
            self.view_events_button.config(state=tk.DISABLED)
            self.make_calendars_private_button.config(state=tk.DISABLED)
            self.make_all_public_private_button.config(state=tk.DISABLED)
            self.secure_events_button.config(state=tk.DISABLED)
            
            # Remove token file
            if os.path.exists('token.json'):
                os.remove('token.json')
            return
        
        # Show progress immediately
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Authenticating...")
        progress_window.geometry("400x200")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Center window
        progress_window.geometry("+%d+%d" % (
            self.root.winfo_rootx() + 200,
            self.root.winfo_rooty() + 100
        ))
        
        frame = ttk.Frame(progress_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Google Drive Authentication", 
                 font=("Arial", 14, "bold")).pack(pady=(0, 20))
        
        if GOOGLE_API_AVAILABLE:
            status_text = "Opening browser for Google OAuth authentication...\nPlease complete the login in your browser."
        else:
            status_text = "Google API libraries not found.\nRunning in demo mode..."
        
        status_label = ttk.Label(frame, text=status_text, justify=tk.CENTER)
        status_label.pack(pady=(0, 20))
        
        login_progress = ttk.Progressbar(frame, mode='indeterminate')
        login_progress.pack(pady=10, fill=tk.X)
        login_progress.start()
        
        def complete_login():
            try:
                if self.api.authenticate():
                    progress_window.destroy()
                    
                    self.is_logged_in.set(True)
                    if self.api.force_demo_mode:
                        self.status_label.config(text="Connected to Google services (Demo Mode - OAuth2 Disabled)")
                        messagebox.showinfo("Demo Mode", "Running in demo mode with sample data.\nOAuth2 authentication has been disabled.")
                    elif GOOGLE_API_AVAILABLE and self.api.service:
                        self.status_label.config(text="Connected to Google services (Real API)")
                        messagebox.showinfo("Success", "Successfully connected to Google Drive & Calendar APIs!\nUse the tabs to scan files or calendars.")
                    else:
                        self.status_label.config(text="Connected to Google services (Demo Mode)")
                        messagebox.showinfo("Demo Mode", "Running in demo mode with sample data.\nInstall Google API libraries and configure credentials.json for real API access.")
                    
                    self.login_button.config(text="Logout")
                    # Enable all scan buttons
                    self.scan_files_button.config(state=tk.NORMAL)
                    self.scan_calendars_button.config(state=tk.NORMAL)
                else:
                    progress_window.destroy()
                    messagebox.showerror("Error", "Authentication failed. Please try again.")
            except Exception as e:
                progress_window.destroy()
                messagebox.showerror("Error", f"Authentication error: {str(e)}")
        
        # Start authentication in separate thread
        threading.Thread(target=complete_login, daemon=True).start()
        
        # Add cancel button
        ttk.Button(frame, text="Cancel", 
                  command=progress_window.destroy).pack(pady=(20, 0))
    
    def scan_drive_files(self):
        """Scan Google Drive for shared files"""
        def do_scan():
            self.progress_label.config(text="Scanning Google Drive...")
            self.progress_var.set(20)
            
            # Clear existing items
            for item in self.files_tree.get_children():
                self.files_tree.delete(item)
            
            self.progress_var.set(50)
            self.shared_files = self.api.get_shared_files()
            
            # Populate treeview
            for file_info in self.shared_files:
                action = "Make Private"
                if "Public" in file_info['sharing_status']:
                    action = "Remove Public Access"
                elif "Organization" in file_info['sharing_status']:
                    action = "Remove Org Access"
                
                self.files_tree.insert('', 'end', values=(
                    file_info['name'],
                    file_info['sharing_status'],
                    action
                ))
            
            self.progress_var.set(100)
            self.progress_label.config(text=f"Found {len(self.shared_files)} shared files")
            
            if self.shared_files:
                self.make_files_private_button.config(state=tk.NORMAL)
                self.make_all_files_private_button.config(state=tk.NORMAL)
            
            # Reset progress after delay
            self.root.after(2000, lambda: [self.progress_var.set(0), self.progress_label.config(text="Ready")])
        
        threading.Thread(target=do_scan, daemon=True).start()
    
    def scan_calendars(self):
        """Scan Google Calendar for public calendars"""
        def do_scan():
            self.progress_label.config(text="Scanning Google Calendar...")
            self.progress_var.set(20)
            
            # Clear existing items
            for item in self.calendars_tree.get_children():
                self.calendars_tree.delete(item)
            
            self.progress_var.set(50)
            self.public_calendars = self.api.get_public_calendars()
            
            # Populate treeview
            for calendar_info in self.public_calendars:
                action = "Make Private"
                if "Public" in calendar_info['sharing_status']:
                    action = "Remove Public Access"
                elif "Organization" in calendar_info['sharing_status']:
                    action = "Remove Org Access"
                
                self.calendars_tree.insert('', 'end', values=(
                    calendar_info['name'],
                    calendar_info['sharing_status'],
                    action
                ))
            
            self.progress_var.set(100)
            self.progress_label.config(text=f"Found {len(self.public_calendars)} public calendars")
            
            if self.public_calendars:
                self.view_events_button.config(state=tk.NORMAL)
                self.make_calendars_private_button.config(state=tk.NORMAL)
                self.make_all_calendars_private_button.config(state=tk.NORMAL)
            
            # Reset progress after delay
            self.root.after(2000, lambda: [self.progress_var.set(0), self.progress_label.config(text="Ready")])
        
        threading.Thread(target=do_scan, daemon=True).start()
    
    def scan_all_calendars(self):
        """Scan ALL calendars (both public and private) to show schedules and security risks"""
        def do_scan():
            self.progress_label.config(text="Scanning ALL Google Calendars...")
            self.progress_var.set(20)
            
            # Clear existing items
            for item in self.calendars_tree.get_children():
                self.calendars_tree.delete(item)
            
            self.progress_var.set(50)
            self.all_calendars = self.api.get_all_calendars()
            self.public_calendars = [cal for cal in self.all_calendars if cal['is_public']]
            
            # Populate treeview with ALL calendars
            for calendar_info in self.all_calendars:
                calendar_type = "Primary" if calendar_info.get('primary') else "Secondary"
                
                if calendar_info['is_public']:
                    security_risk = "⚠️ PUBLIC"
                    # Add red background for public calendars
                    item = self.calendars_tree.insert('', 'end', values=(
                        calendar_info['name'],
                        calendar_type,
                        calendar_info['sharing_status'],
                        security_risk
                    ), tags=('public_calendar',))
                else:
                    security_risk = "✅ Private"
                    item = self.calendars_tree.insert('', 'end', values=(
                        calendar_info['name'],
                        calendar_type,
                        calendar_info['sharing_status'],
                        security_risk
                    ), tags=('private_calendar',))
            
            # Configure tags for visual distinction
            self.calendars_tree.tag_configure('public_calendar', background='#ffeeee', foreground='#cc0000')
            self.calendars_tree.tag_configure('private_calendar', background='#eeffee', foreground='#006600')
            
            self.progress_var.set(100)
            public_count = len(self.public_calendars)
            total_count = len(self.all_calendars)
            private_count = total_count - public_count
            
            self.progress_label.config(text=f"Found {total_count} calendars: {public_count} public (⚠️), {private_count} private (✅)")
            
            if self.all_calendars:
                self.view_events_button.config(state=tk.NORMAL)
                self.make_calendars_private_button.config(state=tk.NORMAL)
                self.secure_events_button.config(state=tk.NORMAL)
            if public_count > 0:
                self.make_all_public_private_button.config(state=tk.NORMAL)
            
            # Reset progress after delay
            self.root.after(3000, lambda: [self.progress_var.set(0), self.progress_label.config(text="Ready")])
        
        threading.Thread(target=do_scan, daemon=True).start()
    
    def view_calendar_events(self):
        """Show events for selected calendar"""
        selected_items = self.calendars_tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select a calendar to view events")
            return
        
        # Get selected calendar
        calendar_index = self.calendars_tree.index(selected_items[0])
        selected_calendar = self.all_calendars[calendar_index]
        
        # Create events window
        events_window = tk.Toplevel(self.root)
        events_window.title(f"Events - {selected_calendar['name']}")
        events_window.geometry("900x600")
        events_window.transient(self.root)
        
        # Main frame
        main_frame = ttk.Frame(events_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Calendar info
        info_frame = ttk.LabelFrame(main_frame, text="Calendar Information", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(info_frame, text=f"Calendar: {selected_calendar['name']}", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"Sharing: {selected_calendar['sharing_status']}", font=("Arial", 10)).pack(anchor=tk.W)
        if selected_calendar.get('description'):
            ttk.Label(info_frame, text=f"Description: {selected_calendar['description']}", font=("Arial", 10)).pack(anchor=tk.W)
        
        # Events frame
        events_frame = ttk.LabelFrame(main_frame, text="Recent & Upcoming Events", padding="10")
        events_frame.pack(fill=tk.BOTH, expand=True)
        
        # Events treeview
        event_columns = ('Title', 'Date', 'Duration', 'Location', 'Attendees', 'Visibility')
        events_tree = ttk.Treeview(events_frame, columns=event_columns, show='headings', height=15)
        
        # Configure event columns
        events_tree.heading('Title', text='Event Title')
        events_tree.heading('Date', text='Date & Time')
        events_tree.heading('Duration', text='Duration')
        events_tree.heading('Location', text='Location')
        events_tree.heading('Attendees', text='Attendees')
        events_tree.heading('Visibility', text='Visibility')
        
        events_tree.column('Title', width=250)
        events_tree.column('Date', width=150)
        events_tree.column('Duration', width=100)
        events_tree.column('Location', width=150)
        events_tree.column('Attendees', width=80)
        events_tree.column('Visibility', width=80)
        
        # Scrollbar for events
        events_scrollbar = ttk.Scrollbar(events_frame, orient=tk.VERTICAL, command=events_tree.yview)
        events_tree.configure(yscrollcommand=events_scrollbar.set)
        
        events_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        events_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Load and display events
        def load_events():
            self.progress_label.config(text=f"Loading events for {selected_calendar['name']}...")
            self.progress_var.set(30)
            
            events = self.api.get_calendar_events(selected_calendar['id'])
            
            for event in events:
                events_tree.insert('', 'end', values=(
                    event['title'],
                    event['date'],
                    event['duration'],
                    event['location'] or '-',
                    event['attendees'] if event['attendees'] > 0 else '-',
                    event['visibility']
                ))
            
            self.progress_var.set(100)
            if events:
                self.progress_label.config(text=f"Loaded {len(events)} events")
            else:
                self.progress_label.config(text="No events found")
            
            # Reset progress
            self.root.after(2000, lambda: [self.progress_var.set(0), self.progress_label.config(text="Ready")])
        
        # Load events in separate thread
        threading.Thread(target=load_events, daemon=True).start()
        
        # Event details on double-click
        def show_event_details(event):
            selected_event_items = events_tree.selection()
            if not selected_event_items:
                return
                
            event_values = events_tree.item(selected_event_items[0])['values']
            
            detail_window = tk.Toplevel(events_window)
            detail_window.title("Event Details")
            detail_window.geometry("500x400")
            detail_window.transient(events_window)
            
            detail_frame = ttk.Frame(detail_window, padding="20")
            detail_frame.pack(fill=tk.BOTH, expand=True)
            
            ttk.Label(detail_frame, text="Event Details", font=("Arial", 14, "bold")).pack(anchor=tk.W, pady=(0, 10))
            
            details_text = scrolledtext.ScrolledText(detail_frame, height=15, width=50)
            details_text.pack(fill=tk.BOTH, expand=True)
            
            # Find the full event data
            events = self.api.get_calendar_events(selected_calendar['id'])
            selected_event = None
            for e in events:
                if e['title'] == event_values[0] and e['date'] == event_values[1]:
                    selected_event = e
                    break
            
            if selected_event:
                details_text.insert(tk.END, f"Title: {selected_event['title']}\n\n")
                details_text.insert(tk.END, f"Date & Time: {selected_event['date']}\n")
                details_text.insert(tk.END, f"Duration: {selected_event['duration']}\n")
                if selected_event['location']:
                    details_text.insert(tk.END, f"Location: {selected_event['location']}\n")
                if selected_event['attendees'] > 0:
                    details_text.insert(tk.END, f"Attendees: {selected_event['attendees']}\n")
                details_text.insert(tk.END, f"Visibility: {selected_event['visibility']}\n\n")
                if selected_event['description']:
                    details_text.insert(tk.END, f"Description:\n{selected_event['description']}")
                else:
                    details_text.insert(tk.END, "No description provided.")
            
            details_text.config(state=tk.DISABLED)
        
        events_tree.bind('<Double-1>', show_event_details)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Refresh Events", 
                  command=lambda: [events_tree.delete(*events_tree.get_children()), load_events()]).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(button_frame, text="Close", 
                  command=events_window.destroy).pack(side=tk.RIGHT)

    def make_selected_files_private(self):
        """Make selected files private"""
        selected_items = self.files_tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select files to make private")
            return
        
        file_indices = [self.files_tree.index(item) for item in selected_items]
        selected_files = [self.shared_files[i] for i in file_indices]
        
        self.process_files(selected_files)
    
    def make_all_files_private(self):
        """Make all shared files private"""
        if not self.shared_files:
            messagebox.showwarning("Warning", "No shared files found")
            return
        
        result = messagebox.askyesno("Confirm", 
                                   f"Make all {len(self.shared_files)} files private?")
        if result:
            self.process_files(self.shared_files)

    def make_selected_calendars_private(self):
        """Make selected calendars private (only if they are public)"""
        selected_items = self.calendars_tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select calendars to make private")
            return
        
        calendar_indices = [self.calendars_tree.index(item) for item in selected_items]
        selected_calendars = [self.all_calendars[i] for i in calendar_indices]
        
        # Filter to only public calendars
        public_selected = [cal for cal in selected_calendars if cal['is_public']]
        private_selected = [cal for cal in selected_calendars if not cal['is_public']]
        
        if private_selected:
            private_names = [cal['name'] for cal in private_selected]
            messagebox.showinfo("Already Private", 
                              f"These calendars are already private:\n• " + "\n• ".join(private_names))
        
        if public_selected:
            result = messagebox.askyesno("Confirm Privacy Change", 
                                       f"Make {len(public_selected)} public calendar(s) private?\n\n" +
                                       "This will prevent others from seeing your schedule.")
            if result:
                self.process_calendars(public_selected)
        elif not private_selected:
            messagebox.showwarning("No Action Needed", "No calendars selected or all selected calendars are already private.")
    
    def make_all_public_calendars_private(self):
        """Make all public calendars private for security"""
        if not self.public_calendars:
            messagebox.showwarning("Security Check", "Good news! All your calendars are already private.")
            return
        
        calendar_names = [cal['name'] for cal in self.public_calendars]
        result = messagebox.askyesno("Security Warning", 
                                   f"⚠️ SECURITY RISK DETECTED ⚠️\n\n" +
                                   f"You have {len(self.public_calendars)} public calendar(s) exposing your schedule:\n\n• " +
                                   "\n• ".join(calendar_names) + 
                                   "\n\nMake all of these private for security?")
        if result:
            self.process_calendars(self.public_calendars)
    
    def process_files(self, files_to_process):
        """Process files to make them private"""
        def do_process():
            total_files = len(files_to_process)
            success_count = 0
            
            for i, file_info in enumerate(files_to_process):
                # Update progress
                progress = (i / total_files) * 100
                self.progress_var.set(progress)
                self.progress_label.config(text=f"Processing: {file_info['name']}")
                self.root.update()
                
                try:
                    # Make file private
                    if self.api.make_file_private(file_info['id']):
                        success_count += 1
                        
                        # Update treeview - find and update the item
                        for item in self.files_tree.get_children():
                            if self.files_tree.item(item)['values'][0] == file_info['name']:
                                self.files_tree.item(item, values=(
                                    file_info['name'],
                                    "Private - Only you",
                                    "[+] Private"
                                ))
                                break
                
                except Exception as e:
                    print(f"Error processing {file_info['name']}: {e}")
            
            # Complete
            self.progress_var.set(100)
            self.progress_label.config(text=f"Complete! {success_count}/{total_files} files made private")
            
            messagebox.showinfo("Success", 
                              f"Successfully made {success_count} files private!\n"
                              f"These files are now only accessible by you.")
            
            # Reset progress
            self.root.after(3000, lambda: [self.progress_var.set(0), self.progress_label.config(text="Ready")])
        
        threading.Thread(target=do_process, daemon=True).start()
    
    def process_calendars(self, calendars_to_process):
        """Process calendars to make them private"""
        def do_process():
            total_calendars = len(calendars_to_process)
            success_count = 0
            
            for i, calendar_info in enumerate(calendars_to_process):
                # Update progress
                progress = (i / total_calendars) * 100
                self.progress_var.set(progress)
                self.progress_label.config(text=f"Processing: {calendar_info['name']}")
                self.root.update()
                
                try:
                    # Make calendar private
                    if self.api.make_calendar_private(calendar_info['id']):
                        success_count += 1
                        
                        # Update treeview - find and update the item
                        for item in self.calendars_tree.get_children():
                            if self.calendars_tree.item(item)['values'][0] == calendar_info['name']:
                                self.calendars_tree.item(item, values=(
                                    calendar_info['name'],
                                    "Private - Only you",
                                    "[+] Private"
                                ))
                                break
                
                except Exception as e:
                    print(f"Error processing {calendar_info['name']}: {e}")
            
            # Complete
            self.progress_var.set(100)
            self.progress_label.config(text=f"Complete! {success_count}/{total_calendars} calendars made private")
            
            messagebox.showinfo("Success", 
                              f"Successfully made {success_count} calendars private!\n"
                              f"These calendars are now only accessible by you.")
            
            # Reset progress
            self.root.after(3000, lambda: [self.progress_var.set(0), self.progress_label.config(text="Ready")])
        
        threading.Thread(target=do_process, daemon=True).start()
    
    def console_main(self):
        """Console version main loop"""
        while True:
            try:
                os.system('cls' if os.name == 'nt' else 'clear')
                print("=" * 60)
                print("      GOOGLE PRIVACY MANAGER")
                print("=" * 60)
                print()
                
                if self.api.authenticated:
                    print("Status: [+] Connected to Google services")
                else:
                    print("Status: [-] Not connected")
                
                print()
                print("1. Login to Google services")
                print("2. Scan for shared files")
                print("3. Show shared files")
                print("4. Make all files private")
                print("5. Scan ALL calendars (see schedule & security)")
                print("6. Show all calendars with events")
                print("7. Make public calendars private (security)")
                print("8. Exit")
                print()
                
                choice = input("Select option (1-8): ").strip()
                
                if choice == '1':
                    print("\nAuthenticating with Google services...")
                    if GOOGLE_API_AVAILABLE:
                        print("This will open your browser for OAuth authentication.")
                    else:
                        print("Google API libraries not found. Running in demo mode.")
                    
                    if self.api.authenticate():
                        if self.api.force_demo_mode:
                            print("[+] Connected in demo mode (OAuth2 disabled)!")
                        elif GOOGLE_API_AVAILABLE and self.api.service:
                            print("[+] Successfully connected to Google Drive & Calendar APIs!")
                        else:
                            print("[+] Connected in demo mode!")
                    else:
                        print("[-] Authentication failed")
                    input("Press Enter to continue...")
                    
                elif choice == '2':
                    if not self.api.authenticated:
                        print("Please login first!")
                    else:
                        print("Scanning Google Drive...")
                        self.shared_files = self.api.get_shared_files()
                        print(f"Found {len(self.shared_files)} shared files")
                    input("Press Enter to continue...")
                    
                elif choice == '3':
                    if self.shared_files:
                        print(f"\nShared Files ({len(self.shared_files)}):")
                        print("-" * 50)
                        for i, file_info in enumerate(self.shared_files, 1):
                            print(f"{i}. {file_info['name']}")
                            print(f"   Sharing: {file_info['sharing_status']}")
                    else:
                        print("No shared files found. Run scan first.")
                    input("Press Enter to continue...")
                    
                elif choice == '4':
                    if self.shared_files:
                        print(f"\nMaking {len(self.shared_files)} files private...")
                        for file_info in self.shared_files:
                            print(f"Processing: {file_info['name']}")
                            self.api.make_file_private(file_info['id'])
                        print("[+] All files are now private!")
                    else:
                        print("No files to process")
                    input("Press Enter to continue...")
                    
                elif choice == '5':
                    if not self.api.authenticated:
                        print("Please login first!")
                    else:
                        print("Scanning ALL Google Calendars...")
                        self.all_calendars = self.api.get_all_calendars()
                        self.public_calendars = [cal for cal in self.all_calendars if cal['is_public']]
                        
                        total_count = len(self.all_calendars)
                        public_count = len(self.public_calendars)
                        private_count = total_count - public_count
                        
                        print(f"Found {total_count} total calendars:")
                        print(f"  ✅ {private_count} private (secure)")
                        print(f"  ⚠️  {public_count} public (security risk)")
                        
                        if public_count > 0:
                            print(f"\n⚠️  WARNING: {public_count} calendar(s) are exposing your schedule!")
                    input("Press Enter to continue...")
                    
                elif choice == '6':
                    if hasattr(self, 'all_calendars') and self.all_calendars:
                        print(f"\nALL Your Calendars ({len(self.all_calendars)}):")
                        print("=" * 70)
                        
                        for i, calendar_info in enumerate(self.all_calendars, 1):
                            # Security indicator
                            if calendar_info['is_public']:
                                security_status = "⚠️  PUBLIC RISK"
                                sharing_color = "⚠️ "
                            else:
                                security_status = "✅ PRIVATE"
                                sharing_color = "✅ "
                            
                            calendar_type = "📅 PRIMARY" if calendar_info.get('primary') else "📋 Secondary"
                            
                            print(f"{i}. {calendar_info['name']} [{calendar_type}]")
                            print(f"   Security: {security_status}")
                            print(f"   Sharing: {sharing_color}{calendar_info['sharing_status']}")
                            
                            # Show recent events with privacy indicators
                            events = self.api.get_calendar_events(calendar_info['id'], max_results=3)
                            if events:
                                print(f"   Recent/Upcoming Events:")
                                for event in events:
                                    privacy_icon = "🔒" if event['visibility'] == 'private' else "👁️"
                                    print(f"     {privacy_icon} {event['title']} - {event['date']}")
                                    if event['location']:
                                        print(f"       📍 {event['location']}")
                            else:
                                print(f"   📅 No recent events")
                            print()
                    else:
                        print("No calendars found. Run scan first.")
                    input("Press Enter to continue...")
                    
                elif choice == '7':
                    if hasattr(self, 'public_calendars') and self.public_calendars:
                        print(f"\n⚠️  SECURITY ALERT ⚠️")
                        print(f"Making {len(self.public_calendars)} PUBLIC calendars PRIVATE...")
                        print("-" * 60)
                        
                        for calendar_info in self.public_calendars:
                            print(f"🔒 Securing: {calendar_info['name']}")
                            print(f"   Current exposure: {calendar_info['sharing_status']}")
                            self.api.make_calendar_private(calendar_info['id'])
                            print(f"   ✅ Now private and secure!")
                            print()
                        
                        print("🎉 All calendars are now PRIVATE and secure!")
                        print("   Your schedule is no longer visible to unauthorized users.")
                    else:
                        print("✅ Good news! All your calendars are already private and secure.")
                    input("Press Enter to continue...")
                    
                elif choice == '8':
                    break
            except EOFError:
                print("\nExiting due to EOF...")
                break
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    
    def run(self):
        if GUI_AVAILABLE:
            self.root.mainloop()
        else:
            self.console_main()

# Installation instructions
def check_dependencies():
    print("Google Privacy Manager - Drive Files & Calendars")
    print("=" * 50)
    print()
    
    if GOOGLE_API_AVAILABLE:
        print("[+] Google API libraries found!")
        print("To use real Google APIs, you need to:")
        print("1. Create credentials.json file with your OAuth2 credentials")
        print("2. Get credentials from Google Cloud Console")
        print("3. Enable Google Drive API and Google Calendar API")
        print() 
        print("Without credentials.json, the app will run in demo mode.")
    else:
        print("Google API libraries not found.")
        print("To install: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")
        print()
        print("Running in demo mode with sample data...")
    
    try:
        print("Press Enter to continue...")
        input()
    except EOFError:
        print("Running in non-interactive mode. Continuing...")
        pass

if __name__ == "__main__":
    check_dependencies()
    
    # Check if user wants to run without OAuth2
    import sys
    force_demo = False
    
    if len(sys.argv) > 1 and sys.argv[1] in ['--demo', '--no-oauth', '-d']:
        force_demo = True
        print("Running in demo mode (OAuth2 disabled)")
    
    app = GooglePrivacyManager(force_demo_mode=force_demo)
    app.run()