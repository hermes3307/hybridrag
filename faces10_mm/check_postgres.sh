#!/bin/bash
# PostgreSQL Diagnostic Script

echo "=== PostgreSQL Diagnostic Report ==="
echo ""

echo "1. PostgreSQL Service Status:"
systemctl is-active postgresql
systemctl status postgresql | grep "Active:"
echo ""

echo "2. PostgreSQL Version:"
psql --version
echo ""

echo "3. Listening Ports:"
sudo netstat -tlnp | grep postgres || ss -tlnp | grep postgres
echo ""

echo "4. Database Connection Test (using .env settings):"
PGPASSWORD=postgres psql -U postgres -h localhost -d vector_db -c "SELECT 1 as connected;" 2>&1 || echo "Connection failed"
echo ""

echo "5. Face count in database:"
PGPASSWORD=postgres psql -U postgres -h localhost -d vector_db -c "SELECT COUNT(*) as face_count FROM faces;" 2>&1 || echo "Query failed"
echo ""

echo "6. Check if pg_hba.conf allows connections:"
echo "   (Requires sudo access to view)"
echo ""

echo "7. Environment variables:"
env | grep POSTGRES
echo ""

echo "8. Database configuration from .env and defaults:"
python3 << 'EOF'
import os
print(f"POSTGRES_HOST: {os.getenv('POSTGRES_HOST', 'localhost')}")
print(f"POSTGRES_PORT: {os.getenv('POSTGRES_PORT', '5432')}")
print(f"POSTGRES_DB: {os.getenv('POSTGRES_DB', 'vector_db')}")
print(f"POSTGRES_USER: {os.getenv('POSTGRES_USER', 'postgres')}")
print(f"POSTGRES_PASSWORD: {'*' * len(os.getenv('POSTGRES_PASSWORD', 'postgres'))}")
EOF

echo ""
echo "=== Suggested Fixes ==="
echo ""
echo "If authentication fails, try one of these:"
echo "1. Edit /etc/postgresql/16/main/pg_hba.conf (requires sudo)"
echo "   Change 'peer' to 'md5' for local connections"
echo "   sudo nano /etc/postgresql/16/main/pg_hba.conf"
echo ""
echo "2. Reload PostgreSQL after changes:"
echo "   sudo systemctl reload postgresql"
echo ""
echo "3. Test connection with password:"
echo "   PGPASSWORD=postgres psql -U postgres -h localhost -d face_recognition"
echo ""
