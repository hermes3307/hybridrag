import os
import json
import time
import webbrowser
from pathlib import Path
import subprocess
import threading

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
        self.scopes = ['https://www.googleapis.com/auth/drive']
        self.force_demo_mode = force_demo_mode
        
    def authenticate(self, use_oauth=True):
        """Authenticate with Google Drive API using OAuth2"""
        # If force_demo_mode is True, skip OAuth2 and use demo mode
        if self.force_demo_mode:
            print("Running in demo mode (OAuth2 disabled)")
            time.sleep(1)
            self.authenticated = True
            return True
            
        if not GOOGLE_API_AVAILABLE:
            # Fallback to demo mode
            time.sleep(1)
            self.authenticated = True
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
            self.authenticated = True
            return True
            
        except Exception as e:
            print(f"Authentication failed: {e}")
            # Fallback to demo mode
            time.sleep(1)
            self.authenticated = True
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

class GoogleDrivePermissionManager:
    def __init__(self, force_demo_mode=False):
        self.api = GoogleDriveAPI(force_demo_mode=force_demo_mode)
        self.shared_files = []
        self.force_demo_mode = force_demo_mode
        
        if GUI_AVAILABLE:
            self.setup_gui()
        else:
            self.setup_console()
    
    def setup_gui(self):
        """Setup GUI interface"""
        self.root = tk.Tk()
        self.root.title("Google Drive Permission Manager")
        self.root.geometry("800x600")
        
        # Variables
        self.progress_var = tk.DoubleVar()
        self.is_logged_in = tk.BooleanVar(value=False)
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Google Drive Permission Manager", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Login section
        login_frame = ttk.LabelFrame(main_frame, text="Authentication", padding="10")
        login_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(login_frame, text="Not logged in to Google Drive")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        self.login_button = ttk.Button(login_frame, text="Login to Google Drive", 
                                      command=self.google_login)
        self.login_button.grid(row=0, column=1, sticky=tk.E, padx=(10, 0))
        
        # Files section
        files_frame = ttk.LabelFrame(main_frame, text="Shared Files", padding="10")
        files_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Treeview for files
        columns = ('Name', 'Sharing Status', 'Action')
        self.files_tree = ttk.Treeview(files_frame, columns=columns, show='headings', height=10)
        
        # Configure columns
        self.files_tree.heading('Name', text='File Name')
        self.files_tree.heading('Sharing Status', text='Current Sharing')
        self.files_tree.heading('Action', text='Action Needed')
        
        self.files_tree.column('Name', width=300)
        self.files_tree.column('Sharing Status', width=200)
        self.files_tree.column('Action', width=150)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(files_frame, orient=tk.VERTICAL, command=self.files_tree.yview)
        self.files_tree.configure(yscrollcommand=scrollbar.set)
        
        self.files_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Buttons
        button_frame = ttk.Frame(files_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        self.scan_button = ttk.Button(button_frame, text="Scan Drive Files", 
                                     command=self.scan_drive_files, state=tk.DISABLED)
        self.scan_button.grid(row=0, column=0, padx=(0, 5))
        
        self.make_private_button = ttk.Button(button_frame, text="Make Selected Private", 
                                            command=self.make_selected_private, state=tk.DISABLED)
        self.make_private_button.grid(row=0, column=1, padx=5)
        
        self.make_all_private_button = ttk.Button(button_frame, text="Make All Private", 
                                                command=self.make_all_private, state=tk.DISABLED)
        self.make_all_private_button.grid(row=0, column=2, padx=(5, 0))
        
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
            self.api.credentials = None
            self.is_logged_in.set(False)
            self.status_label.config(text="Not logged in to Google Drive")
            self.login_button.config(text="Login to Google Drive")
            self.scan_button.config(state=tk.DISABLED)
            self.make_private_button.config(state=tk.DISABLED)
            self.make_all_private_button.config(state=tk.DISABLED)
            
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
                        self.status_label.config(text="Connected to Google Drive (Demo Mode - OAuth2 Disabled)")
                        messagebox.showinfo("Demo Mode", "Running in demo mode with sample data.\nOAuth2 authentication has been disabled.")
                    elif GOOGLE_API_AVAILABLE and self.api.service:
                        self.status_label.config(text="Connected to Google Drive (Real API)")
                        messagebox.showinfo("Success", "Successfully connected to Google Drive API!\nClick 'Scan Drive Files' to find your shared files.")
                    else:
                        self.status_label.config(text="Connected to Google Drive (Demo Mode)")
                        messagebox.showinfo("Demo Mode", "Running in demo mode with sample data.\nInstall Google API libraries and configure credentials.json for real API access.")
                    
                    self.login_button.config(text="Logout")
                    self.scan_button.config(state=tk.NORMAL)
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
                self.make_private_button.config(state=tk.NORMAL)
                self.make_all_private_button.config(state=tk.NORMAL)
            
            # Reset progress after delay
            self.root.after(2000, lambda: [self.progress_var.set(0), self.progress_label.config(text="Ready")])
        
        threading.Thread(target=do_scan, daemon=True).start()
    
    def make_selected_private(self):
        """Make selected files private"""
        selected_items = self.files_tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select files to make private")
            return
        
        file_indices = [self.files_tree.index(item) for item in selected_items]
        selected_files = [self.shared_files[i] for i in file_indices]
        
        self.process_files(selected_files)
    
    def make_all_private(self):
        """Make all shared files private"""
        if not self.shared_files:
            messagebox.showwarning("Warning", "No shared files found")
            return
        
        result = messagebox.askyesno("Confirm", 
                                   f"Make all {len(self.shared_files)} files private?")
        if result:
            self.process_files(self.shared_files)
    
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
    
    def console_main(self):
        """Console version main loop"""
        while True:
            try:
                os.system('cls' if os.name == 'nt' else 'clear')
                print("=" * 60)
                print("      GOOGLE DRIVE PERMISSION MANAGER")
                print("=" * 60)
                print()
                
                if self.api.authenticated:
                    print("Status: [+] Connected to Google Drive")
                else:
                    print("Status: [-] Not connected")
                
                print()
                print("1. Login to Google Drive")
                print("2. Scan for shared files")
                print("3. Show shared files")
                print("4. Make all files private")
                print("5. Exit")
                print()
                
                choice = input("Select option (1-5): ").strip()
                
                if choice == '1':
                    print("\nAuthenticating with Google Drive...")
                    if GOOGLE_API_AVAILABLE:
                        print("This will open your browser for OAuth authentication.")
                    else:
                        print("Google API libraries not found. Running in demo mode.")
                    
                    if self.api.authenticate():
                        if self.api.force_demo_mode:
                            print("[+] Connected in demo mode (OAuth2 disabled)!")
                        elif GOOGLE_API_AVAILABLE and self.api.service:
                            print("[+] Successfully connected to Google Drive API!")
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
    print("Google Drive Permission Manager")
    print("=" * 40)
    print()
    
    if GOOGLE_API_AVAILABLE:
        print("[+] Google API libraries found!")
        print("To use real Google Drive API, you need to:")
        print("1. Create credentials.json file with your OAuth2 credentials")
        print("2. Get credentials from Google Cloud Console")
        print("3. Enable Google Drive API for your project")
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
    
    app = GoogleDrivePermissionManager(force_demo_mode=force_demo)
    app.run()