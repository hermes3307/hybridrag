# Ollama WSL2 Connection Setup

## Problem

You have Ollama installed on Windows but need to connect from WSL2.

## Solution

By default, Ollama only listens on `localhost` (127.0.0.1), which WSL2 cannot access. We need to configure it to listen on all interfaces.

## Step-by-Step Fix

### Option 1: Configure Ollama to Listen on All Interfaces (Recommended)

**Step 1: Stop Ollama (if running)**
- Open Task Manager (Ctrl+Shift+Esc)
- Find "Ollama" process
- Right-click → End Task

**Step 2: Set Environment Variable in Windows**

Open PowerShell as Administrator and run:

```powershell
[System.Environment]::SetEnvironmentVariable('OLLAMA_HOST', '0.0.0.0:11434', 'User')
```

**Step 3: Restart Ollama**
- Open Ollama from Start Menu
- Or run from PowerShell: `ollama serve`

**Step 4: Verify from Windows**

In PowerShell:
```powershell
curl http://localhost:11434/api/tags
```

You should see a JSON response with your models.

**Step 5: Test from WSL2**

In WSL2 terminal:
```bash
# Your Windows IP (already found)
WINDOWS_IP="172.24.176.1"

# Test connection
curl http://$WINDOWS_IP:11434/api/tags
```

**Step 6: Configure in .env**

Update your `.env` file:
```bash
echo "OLLAMA_HOST=http://172.24.176.1:11434" >> .env
```

### Option 2: Use Windows Firewall Rule

If Option 1 doesn't work, you may need to allow the connection through Windows Firewall:

**Step 1: Open Windows Defender Firewall**
- Press Win+R
- Type: `wf.msc`
- Press Enter

**Step 2: Create Inbound Rule**
- Click "Inbound Rules" → "New Rule"
- Rule Type: Port → Next
- Protocol: TCP
- Specific local ports: 11434 → Next
- Action: Allow the connection → Next
- Profile: Check all → Next
- Name: "Ollama WSL2"→ Finish

**Step 3: Restart Ollama**

### Option 3: Alternative - Use SSH Tunnel (Advanced)

If the above don't work, you can tunnel through SSH:

```bash
# In WSL2
ssh -L 11434:localhost:11434 your_windows_user@172.24.176.1

# Then use
OLLAMA_HOST=http://localhost:11434
```

## Quick Test Commands

### Check if Ollama is running (Windows PowerShell):
```powershell
Get-Process ollama
netstat -an | findstr 11434
```

### Test from WSL2:
```bash
# Test connection
curl http://172.24.176.1:11434/api/tags

# Should return JSON like:
# {"models":[...]}
```

### Pull a model (Windows PowerShell):
```powershell
ollama pull llama3.2
```

### List models (Windows PowerShell):
```powershell
ollama list
```

## Configuration Summary

After successful setup, your `.env` should have:

```bash
# LLM Configuration
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
OLLAMA_HOST=http://172.24.176.1:11434
```

## Verify Setup

Run this script to verify everything:

```bash
#!/bin/bash

echo "=== Ollama WSL2 Connection Test ==="
echo ""

# Get Windows IP
WINDOWS_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
echo "Windows Host IP: $WINDOWS_IP"
echo ""

# Test connection
echo "Testing connection to Ollama..."
if curl -s http://$WINDOWS_IP:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Connection successful!"
    echo ""
    echo "Available models:"
    curl -s http://$WINDOWS_IP:11434/api/tags | python3 -m json.tool 2>/dev/null || echo "  (Could not parse models list)"
else
    echo "✗ Connection failed"
    echo ""
    echo "Troubleshooting:"
    echo "1. Make sure Ollama is running on Windows"
    echo "2. Set OLLAMA_HOST=0.0.0.0:11434 in Windows environment"
    echo "3. Check Windows Firewall settings"
    echo "4. Restart Ollama on Windows"
fi
```

## Common Issues

### Issue 1: "Connection refused"

**Cause**: Ollama not listening on 0.0.0.0

**Fix**: Set `OLLAMA_HOST=0.0.0.0:11434` in Windows environment variables

### Issue 2: "Connection timeout"

**Cause**: Windows Firewall blocking

**Fix**: Add firewall rule (see Option 2 above)

### Issue 3: "No models found"

**Cause**: No models pulled yet

**Fix**: In Windows PowerShell:
```powershell
ollama pull llama3.2
```

### Issue 4: Ollama not starting

**Cause**: Port 11434 already in use

**Fix**:
```powershell
# Find what's using port 11434
netstat -ano | findstr 11434

# Kill the process or change Ollama port
$env:OLLAMA_HOST="0.0.0.0:11435"
```

## Recommended Models for WSL2

Since you're running on Windows with WSL2, consider:

- **llama3.2** (2GB) - Best balance
- **phi** (1.6GB) - Fastest
- **mistral** (4GB) - Better quality

## Testing in the GUI

Once configured:

1. Open the application
2. Go to **Configuration** tab
3. **LLM Assistant Configuration** section:
   - Provider: "Ollama (Local)"
   - Model: "llama3.2" (or whichever you pulled)
4. Click **"Test LLM Connection"**
5. Should see: "✓ Connected to Ollama!"
6. Click **"Apply & Reinitialize"**
7. Go to **LLM Assistant** tab
8. Start chatting!

## Need Help?

If you're still having issues:
1. Check Ollama is running: `Get-Process ollama` (PowerShell)
2. Check port is open: `netstat -an | findstr 11434` (PowerShell)
3. Check firewall: Temporarily disable to test
4. Check .env file has correct OLLAMA_HOST

## Final .env Configuration

```bash
# PostgreSQL Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vector_images
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Application Settings
DB_TYPE=pgvector
EMBEDDING_MODEL=statistical
IMAGES_DIR=./images

# Connection Pool Settings
DB_MIN_CONNECTIONS=1
DB_MAX_CONNECTIONS=10

# Vector Search Settings
DEFAULT_SEARCH_LIMIT=10
DISTANCE_METRIC=cosine

# LLM Assistant Settings
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2

# Ollama Configuration for WSL2
OLLAMA_HOST=http://172.24.176.1:11434
```

Save and restart the application!
