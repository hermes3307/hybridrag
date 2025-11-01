# Face Recognition System - Installation Guide

Complete installation guide for setting up PostgreSQL with pgvector for the Face Recognition System.

## Quick Start

```bash
# Run the automated installer
./install.sh
```

That's it! The script will:
- Install PostgreSQL and pgvector
- Create and configure the database
- Install Python dependencies
- Set up the environment configuration
- Test the installation

---

## Manual Installation

If you prefer to install manually or need to troubleshoot, follow these steps:

### 1. Install PostgreSQL

```bash
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib postgresql-server-dev-all
```

### 2. Install Build Dependencies

```bash
sudo apt-get install -y build-essential git
```

### 3. Install pgvector Extension

```bash
cd /tmp
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### 4. Start PostgreSQL

```bash
sudo service postgresql start
sudo service postgresql status
```

### 5. Create Database

```bash
# Create database
sudo -u postgres psql -c "CREATE DATABASE vector_db;"

# Set password
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"

# Enable pgvector extension
sudo -u postgres psql -d vector_db -c "CREATE EXTENSION vector;"
```

### 6. Create Schema

```bash
sudo -u postgres psql -d vector_db -f schema.sql
```

### 7. Install Python Dependencies

```bash
pip3 install --break-system-packages -r requirements.txt
```

### 8. Configure Environment

Copy the template and edit as needed:

```bash
cp .env.template .env
nano .env
```

### 9. Test Installation

```bash
python3 test_pgvector.py
```

---

## Configuration

### Environment Variables (.env)

```bash
# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vector_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Application Settings
DB_TYPE=pgvector
EMBEDDING_MODEL=statistical
FACES_DIR=./faces

# Connection Pool Settings
DB_MIN_CONNECTIONS=1
DB_MAX_CONNECTIONS=10

# Vector Search Settings
DEFAULT_SEARCH_LIMIT=10
DISTANCE_METRIC=cosine
```

---

## Database Management

Use the `db_manage.sh` script for common database operations:

```bash
# Service Management
./db_manage.sh start      # Start PostgreSQL
./db_manage.sh stop       # Stop PostgreSQL
./db_manage.sh status     # Check status
./db_manage.sh restart    # Restart PostgreSQL

# Database Operations
./db_manage.sh connect    # Connect to database
./db_manage.sh stats      # Show statistics
./db_manage.sh backup     # Create backup
./db_manage.sh restore    # Restore from backup
./db_manage.sh reset      # Reset database (deletes all data!)
./db_manage.sh test       # Test connection
```

---

## Testing

### Run Test Suite

```bash
python3 test_pgvector.py
```

This will test:
- Database connection
- Adding faces (single and batch)
- Vector similarity search
- Metadata filtering
- Hybrid search
- Statistics retrieval

### Manual Database Verification

```bash
# Connect to database
sudo -u postgres psql -d vector_db

# List tables
\dt

# Describe faces table
\d faces

# Check pgvector extension
SELECT * FROM pg_extension WHERE extname = 'vector';

# Count faces
SELECT COUNT(*) FROM faces;

# Check indexes
\di

# Quit
\q
```

---

## Running the Application

### GUI Application

```bash
python3 faces.py
```

### Command-Line Search

```bash
./search_cli.py
```

### Interactive Search Examples

```bash
./search_examples.py
```

### Inspect Database

```bash
python3 inspect_database.py
```

---

## Common Issues and Solutions

### Issue: "no password supplied"

**Solution:**
```bash
# Set password
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"

# Update .env file
nano .env  # Set POSTGRES_PASSWORD=postgres
```

### Issue: "pgvector extension not found"

**Solution:**
```bash
sudo -u postgres psql -d vector_db -c "CREATE EXTENSION vector;"
```

### Issue: "relation 'faces' does not exist"

**Solution:**
```bash
sudo -u postgres psql -d vector_db -f schema.sql
```

### Issue: PostgreSQL won't start

**Solution:**
```bash
# Check logs
sudo tail -f /var/log/postgresql/postgresql-*-main.log

# Try restarting
sudo service postgresql restart

# Check status
sudo service postgresql status
```

### Issue: Permission denied

**Solution:**
```bash
# Make scripts executable
chmod +x install.sh db_manage.sh search_cli.py search_examples.py
```

### Issue: Python package installation fails

**Solution:**
```bash
# Use --break-system-packages flag
pip3 install --break-system-packages -r requirements.txt

# Or use a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## System Requirements

- **OS**: Linux (Ubuntu/Debian recommended, also works on WSL)
- **PostgreSQL**: 12 or higher
- **Python**: 3.7 or higher
- **Disk Space**: At least 500MB free
- **Memory**: At least 2GB RAM recommended

---

## Migrating from ChromaDB

If you have existing data in ChromaDB, use the migration script:

```bash
# Preview migration (dry run)
python3 migrate_to_pgvector.py --dry-run

# Perform migration
python3 migrate_to_pgvector.py

# Custom ChromaDB path
python3 migrate_to_pgvector.py --chroma-path ./my_chroma_db
```

---

## Database Maintenance

### Backup Database

```bash
# Using db_manage.sh
./db_manage.sh backup

# Manual backup
sudo -u postgres pg_dump vector_db | gzip > backup_$(date +%Y%m%d).sql.gz
```

### Restore Database

```bash
# Using db_manage.sh
./db_manage.sh restore

# Manual restore
gunzip -c backup_20251030.sql.gz | sudo -u postgres psql vector_db
```

### Vacuum and Analyze

```bash
sudo -u postgres psql -d vector_db -c "VACUUM ANALYZE faces;"
```

### Check Database Size

```bash
sudo -u postgres psql -d vector_db -c "SELECT pg_size_pretty(pg_database_size('vector_db'));"
```

---

## Performance Tuning

### Adjust Index Parameters

Edit `schema.sql` before running:

```sql
-- For larger datasets (>100K faces)
CREATE INDEX idx_embedding_hnsw_cosine ON faces
USING hnsw (embedding vector_cosine_ops)
WITH (m = 32, ef_construction = 128);

-- For smaller datasets (<10K faces)
CREATE INDEX idx_embedding_hnsw_cosine ON faces
USING hnsw (embedding vector_cosine_ops)
WITH (m = 8, ef_construction = 32);
```

### Adjust Connection Pool

Edit `.env`:

```bash
DB_MIN_CONNECTIONS=2
DB_MAX_CONNECTIONS=20
```

### Use Batch Operations

When adding multiple faces, use batch operations:

```python
# Instead of adding one by one
for face_data in face_list:
    system.db_manager.add_face(face_data)

# Use batch insert
face_tuples = [(face_data, "statistical") for face_data in face_list]
system.db_manager.add_faces_batch(face_tuples, batch_size=100)
```

---

## File Structure

```
faces_pgvector6/
├── install.sh              # Automated installation script
├── db_manage.sh           # Database management helper
├── schema.sql             # Database schema
├── .env.template          # Environment variables template
├── .env                   # Your configuration (created by install.sh)
│
├── pgvector_db.py         # Database manager class
├── core.py                # Core backend
├── faces.py               # GUI application
│
├── search_cli.py          # Command-line search tool
├── search_examples.py     # Interactive search examples
├── advanced_search.py     # Advanced search functions
│
├── test_pgvector.py       # Test suite
├── inspect_database.py    # Database inspection tool
├── migrate_to_pgvector.py # ChromaDB migration tool
│
├── requirements.txt       # Python dependencies
├── INSTALLATION.md        # This file
├── PGVECTOR_README.md     # pgvector implementation details
├── SEARCH_GUIDE.md        # Search functionality guide
└── faces/                 # Directory for face images
```

---

## Next Steps

After installation:

1. **Add face images** to the `faces` directory
2. **Run the application**: `python3 faces.py`
3. **Process faces** using the GUI
4. **Search faces** using the search tools
5. **Explore advanced features** in SEARCH_GUIDE.md

---

## Support

For issues or questions:

1. Check this guide
2. Review PGVECTOR_README.md
3. Run `./db_manage.sh test` to verify setup
4. Check PostgreSQL logs: `/var/log/postgresql/`
5. Verify environment variables in `.env`

---

## Quick Reference

```bash
# Installation
./install.sh

# Start/stop service
./db_manage.sh start
./db_manage.sh stop

# Run application
python3 faces.py

# Search
./search_cli.py

# Test
python3 test_pgvector.py

# Backup
./db_manage.sh backup

# Stats
./db_manage.sh stats
```

---

**Last Updated**: 2025-10-30
**Version**: 1.0
