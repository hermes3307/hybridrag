# Google API Setup Guide for Calendar & Drive Privacy Manager

## Step 1: Create or Select a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Note down your Project ID

## Step 2: Enable Required APIs

You need to enable both APIs:

### Enable Google Drive API:
1. Go to [Google Drive API](https://console.cloud.google.com/apis/library/drive.googleapis.com)
2. Click "Enable"

### Enable Google Calendar API:
1. Go to [Google Calendar API](https://console.cloud.google.com/apis/library/calendar-json.googleapis.com)
2. Click "Enable"

## Step 3: Create OAuth2 Credentials

1. Go to [Credentials page](https://console.cloud.google.com/apis/credentials)
2. Click "Create Credentials" → "OAuth client ID"
3. If prompted, configure OAuth consent screen first:
   - Choose "External" (unless you have G Suite)
   - Fill in App name: "Google Privacy Manager"
   - Add your email as developer contact
   - Add test users (your email) if in testing mode
4. For OAuth client ID:
   - Application type: "Desktop application"
   - Name: "Privacy Manager Desktop"
5. Download the JSON file and rename it to `credentials.json`
6. Place `credentials.json` in the same folder as `file_manager.py`

## Step 4: Required OAuth Scopes

The application uses these scopes (already configured in code):
- `https://www.googleapis.com/auth/drive` - For Drive file management
- `https://www.googleapis.com/auth/calendar` - For Calendar management

## Step 5: OAuth Consent Screen Configuration

### Required Fields:
- **App name:** Google Privacy Manager
- **User support email:** Your email
- **Developer contact:** Your email

### Scopes to Add:
1. Go to OAuth consent screen → Scopes
2. Add these scopes:
   - `../auth/drive` (Google Drive API)
   - `../auth/calendar` (Google Calendar API)

### Test Users:
Add your Gmail account as a test user if the app is in testing mode.

## Step 6: Verification (Optional)

For production use, you may need to verify your app:
1. Submit for verification in OAuth consent screen
2. This process can take several days
3. For personal use, you can skip verification and use the app in testing mode

## Step 7: Running the Application

After setup:
1. Place `credentials.json` in the project folder
2. Run: `python file_manager.py`
3. First time will open browser for OAuth authorization
4. Grant permissions for both Drive and Calendar access
5. The app will create `token.json` for subsequent runs

## Alternative: Environment Variables

Instead of `credentials.json`, you can use environment variables:

```bash
export GOOGLE_CLIENT_ID="your-client-id.apps.googleusercontent.com"
export GOOGLE_CLIENT_SECRET="your-client-secret"
```

## Troubleshooting

### Common Issues:

1. **"API not enabled" error:**
   - Ensure both Drive and Calendar APIs are enabled

2. **"Access blocked" error:**
   - Add your email as test user in OAuth consent screen
   - Or complete app verification process

3. **"Invalid redirect URI" error:**
   - Make sure client type is "Desktop application"

4. **Permission denied errors:**
   - Grant all requested permissions during OAuth flow
   - Check that scopes include both drive and calendar access

### Calendar-Specific Issues:

1. **"Calendar not found" errors:**
   - Some system calendars cannot be modified
   - App will skip inaccessible calendars automatically

2. **"Insufficient permissions" for calendar ACL:**
   - Ensure you're the owner of the calendar
   - Some shared calendars cannot be modified even with write access

## Security Notes

- Keep `credentials.json` and `token.json` secure
- Do not commit these files to version control
- For production deployment, consider using service accounts
- Review OAuth scopes - app only requests minimum necessary permissions

## Testing the Setup

1. Run with demo mode first: `python file_manager.py --demo`
2. Then try real API: `python file_manager.py`
3. Check both file scanning and calendar scanning work
4. Test making items private in both tabs