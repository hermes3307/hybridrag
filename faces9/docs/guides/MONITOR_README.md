# pgvector Database Monitor

A comprehensive real-time GUI application for monitoring your PostgreSQL + pgvector database used for face embeddings.

## Features

### 📊 Real-time Monitoring
- **Overview Tab**: Key metrics with auto-refresh
  - Total faces count
  - Number of vectors (embeddings)
  - Active database connections
  - Database and table sizes
  - Last update timestamp

### 🔌 Connection Monitoring
- **Connections Tab**: View all active database connections
  - Process ID (PID)
  - Connected users
  - Application names
  - Client addresses
  - Connection states
  - Query start times
  - Double-click to view full query details

### 🗂️ Vector Browser
- **Vectors Tab**: Browse and inspect individual vectors
  - Paginated list of all faces in the database
  - Click any face to view:
    - **Image Tab**: View the original face image
    - **Metadata Tab**: Complete metadata including:
      - Face ID, file path, timestamp
      - Image hash
      - Age estimate, gender
      - Brightness, contrast, sharpness
      - Full JSONB metadata
    - **Vector Info Tab**: Embedding information
      - Model used
      - Embedding dimension
      - Vector status

### 📈 Statistics
- **Statistics Tab**: Comprehensive database statistics
  - Total faces and embeddings
  - Database and table sizes
  - Date range (oldest to newest)
  - Embedding models breakdown
  - Timestamp of last update

## Installation

### Prerequisites
```bash
# Python packages
pip install psycopg2-binary pillow python-dotenv

# Or if you have a requirements file
pip install -r requirements.txt
```

### Database Configuration
Make sure your `.env` file is properly configured:

```bash
# PostgreSQL Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vector_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
```

## Usage

### Method 1: Direct Python Execution
```bash
python3 monitor.py
```

### Method 2: Using the Launcher Script
```bash
./run_monitor.sh
```

### Method 3: Background Monitoring
You can run the monitor while your embedding process is running:

Terminal 1 (Embedding):
```bash
python3 faces.py
```

Terminal 2 (Monitoring):
```bash
python3 monitor.py
```

## GUI Layout

### Overview Tab
```
┌─────────────────────────────────────────────┐
│         Database Overview                   │
├─────────────────────────────────────────────┤
│  Total Faces:              150              │
│  Vectors:                  145              │
│  Active Connections:       3                │
│  Database Size:            24 MB            │
│  Table Size:               18 MB            │
│  Last Updated:             14:32:05         │
├─────────────────────────────────────────────┤
│  ☑ Auto-refresh  [Refresh Now]             │
│  Interval: [2] seconds     [Set]            │
└─────────────────────────────────────────────┘
```

### Connections Tab
```
┌──────────────────────────────────────────────────────────┐
│  PID  │  User     │  Application  │  Client  │  State   │
├──────────────────────────────────────────────────────────┤
│ 1234  │ postgres  │ python3       │ local    │ active   │
│ 1235  │ postgres  │ monitor.py    │ local    │ idle     │
└──────────────────────────────────────────────────────────┘
```

### Vectors Tab (Split View)
```
┌─────────────────┬─────────────────────────────────────┐
│  Vector List    │  Vector Details                     │
├─────────────────┼─────────────────────────────────────┤
│ Face ID         │  ┌───────────────────────────────┐ │
│ Model           │  │ [Image] [Metadata] [Vector]  │ │
│ Created         │  ├───────────────────────────────┤ │
│ Vector          │  │                               │ │
├─────────────────┤  │    [Face Image Preview]       │ │
│ face_123...     │  │                               │ │
│ face_124...     │  │                               │ │
│ face_125...     │  └───────────────────────────────┘ │
└─────────────────┴─────────────────────────────────────┘
```

## Controls

### Auto-Refresh
- **Checkbox**: Enable/disable automatic updates
- **Default**: Enabled
- **Interval**: Configurable (0.5 - 60 seconds)
- **Default Interval**: 2 seconds

### Manual Refresh
- Click "Refresh Now" button in Overview tab
- Or use keyboard shortcut: F5 (in respective tabs)

### Navigation
- **Vectors Tab**: Use "◀ Previous" and "Next ▶" buttons to navigate pages
- **Default Page Size**: 50 faces per page

### Detail Viewing
- **Click** any face in the Vectors tab to view details
- **Double-click** any connection to view full query details

## Performance Notes

1. **Auto-Refresh Impact**
   - Lower refresh intervals (< 1 second) may impact database performance
   - Recommended: 2-5 seconds for real-time monitoring
   - Use 10+ seconds for passive monitoring

2. **Connection Pooling**
   - Monitor uses connection pooling (1-5 connections)
   - Minimal impact on database performance

3. **Concurrent Usage**
   - Safe to run while embedding processes are active
   - No conflicts with read/write operations

## Troubleshooting

### "Failed to connect to database"
- Check your `.env` file configuration
- Verify PostgreSQL is running: `sudo systemctl status postgresql`
- Test connection: `psql -h localhost -U postgres -d vector_db`

### "pgvector extension not found"
- Install extension: `CREATE EXTENSION vector;`
- Run schema setup script if available

### Images not displaying
- Verify file paths in database match actual files
- Check file permissions
- Ensure PIL/Pillow is installed correctly

### Slow refresh
- Increase refresh interval
- Check database performance
- Verify network connection (if remote database)

## Tips

1. **Monitoring During Embedding**
   - Open monitor in a separate window
   - Watch vector count increase in real-time
   - Monitor for any errors or connection issues

2. **Inspecting Results**
   - Use Vectors tab to browse recently added faces
   - Sort by creation date to see newest entries
   - Check metadata to verify correct feature extraction

3. **Performance Optimization**
   - Disable auto-refresh when not actively monitoring
   - Use Statistics tab for periodic checks
   - Close monitor when not in use to free connections

## Keyboard Shortcuts

- **F5**: Refresh current tab (planned)
- **Ctrl+Q**: Quit application
- **Tab**: Switch between tabs

## Advanced Features

### Custom Queries
The monitor can be extended to run custom SQL queries. Edit the `DatabaseMonitor` class to add custom query methods.

### Export Data
Add export functionality by extending the GUI with export buttons:
```python
def export_to_csv(self):
    # Export face list or statistics to CSV
    pass
```

### Alerts
Add threshold-based alerts for monitoring:
```python
def check_thresholds(self):
    if vector_count < min_threshold:
        messagebox.showwarning("Alert", "Vector count below threshold")
```

## Credits

Built for the faces_pgvector7 face recognition system.
Uses PostgreSQL with pgvector extension for vector similarity search.

## License

Same as the parent project.
