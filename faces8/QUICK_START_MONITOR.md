# Quick Start Guide - pgvector Database Monitor

## What You Just Got

A **real-time GUI application** to monitor your PostgreSQL + pgvector database with:

✅ **Real-time statistics** - See vector counts update live
✅ **Connection monitoring** - Track active database users and queries
✅ **Vector browser** - Click any vector to see its image and metadata
✅ **Works during processing** - Monitor while embeddings are being created

## Quick Start (3 Steps)

### 1. Test Connection
```bash
cd /home/pi/hybridrag/faces_pgvector7
python3 test_monitor.py
```

If you see "✓ All checks passed!" proceed to step 2.

### 2. Launch Monitor
```bash
python3 monitor.py
```

Or use the launcher:
```bash
./run_monitor.sh
```

### 3. Explore the Interface

**Overview Tab** - See key metrics updating in real-time:
- Total faces: Currently showing **4,248 faces** with vectors
- Database size, connections, and more
- Auto-refreshes every 2 seconds (configurable)

**Connections Tab** - Monitor database activity:
- See all active connections
- View user names and queries
- Double-click any row for full query details

**Vectors Tab** - Browse your data:
- **Left panel**: List of all vectors (paginated)
- **Right panel**: Click any vector to see:
  - 🖼️ **Image tab**: View the original face image
  - 📋 **Metadata tab**: All stored information (age, gender, brightness, etc.)
  - 📊 **Vector Info tab**: Embedding model and dimensions

**Statistics Tab** - Detailed database stats:
- Total counts and sizes
- Date ranges
- Breakdown by embedding model

## Using While Processing

You can monitor in real-time while adding vectors:

**Terminal 1** (your embedding script):
```bash
python3 faces.py
# or your processing script
```

**Terminal 2** (monitor):
```bash
python3 monitor.py
```

Watch the vector count increase in real-time! 🚀

## Configuration

The monitor reads from your `.env` file:
```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vector_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
```

## Useful Features

### Change Refresh Interval
1. Go to **Overview** tab
2. See "Interval (s): [2]"
3. Change to any value between 0.5 and 60 seconds
4. Click **Set**

### Pause Auto-Refresh
- Uncheck "Auto-refresh" checkbox
- Use "Refresh Now" button to update manually
- Useful when inspecting specific data

### Navigate Vector List
- Use **◀ Previous** and **Next ▶** buttons
- Shows 50 faces per page
- Click any face to inspect details

### View Original Images
1. Go to **Vectors** tab
2. Click any face in the left list
3. View image in the **Image** tab on the right
4. See full metadata in **Metadata** tab

## Troubleshooting

### "Failed to connect to database"
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# If stopped, start it
sudo systemctl start postgresql

# Test connection manually
psql -h localhost -U postgres -d vector_db
```

### Images not showing
- Check that file paths in database are correct
- Verify images exist in the `./faces` directory
- Check file permissions

### Slow performance
- Increase refresh interval (5-10 seconds)
- Check database load
- Close other database connections if needed

## Tips & Tricks

1. **Monitoring Downloads**
   - Keep monitor open while running face downloads
   - Watch "Total Faces" count increase
   - Check "Connections" tab to see active processes

2. **Quality Control**
   - Use Vectors tab to randomly sample faces
   - Check metadata for correct feature extraction
   - Verify images match their metadata

3. **Performance Monitoring**
   - Statistics tab shows database growth
   - Connections tab reveals slow queries
   - Use to optimize batch sizes

4. **Debugging**
   - Double-click connections to see full queries
   - Check for stuck or slow operations
   - Monitor during troubleshooting sessions

## What's Monitored

### Real-time Metrics
- Total face count
- Vector count (faces with embeddings)
- Active database connections
- Database size (disk usage)
- Table size

### Per-Vector Information
- Face ID and image hash
- File path (clickable to view image)
- Timestamp (when added)
- Embedding model used
- All metadata fields:
  - Age estimate
  - Gender
  - Brightness, contrast, sharpness
  - Face detection results
  - Demographic info
  - Full JSONB metadata

### Connection Details
- Process ID (PID)
- Connected user
- Application name
- Client address
- Connection state
- Active queries
- Query start time

## System Requirements

- Python 3.7+
- tkinter (usually included with Python)
- psycopg2-binary
- Pillow (PIL)
- python-dotenv
- PostgreSQL 12+ with pgvector extension

## Next Steps

1. **Explore the GUI** - Click around and get familiar
2. **Monitor in real-time** - Run it alongside your processing
3. **Inspect your data** - Browse vectors and check quality
4. **Customize** - Adjust refresh intervals to your needs

## Getting Help

Check the full documentation: `MONITOR_README.md`

For issues:
1. Run `python3 test_monitor.py` to diagnose
2. Check PostgreSQL logs
3. Verify `.env` configuration

## Example Session

```bash
# 1. Start your embedding process
python3 faces.py &

# 2. Launch monitor
python3 monitor.py

# Now you can:
# - Watch vector count increase in Overview tab
# - See your script in Connections tab
# - Browse newly added faces in Vectors tab
# - Check statistics in Statistics tab
```

Enjoy monitoring your pgvector database! 🎉
