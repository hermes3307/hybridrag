#!/bin/bash
# Database Setup Script for Image Search System
# This script creates the image_vector database and applies the schema

set -e  # Exit on error

echo "========================================="
echo "Image Search System - Database Setup"
echo "========================================="
echo ""

# Check if PostgreSQL is running
if ! pg_isready -q; then
    echo "Error: PostgreSQL is not running!"
    echo "Please start PostgreSQL with: sudo systemctl start postgresql"
    exit 1
fi

echo "✓ PostgreSQL is running"

# Database name
DB_NAME="image_vector"
DB_USER="postgres"

# Check if database already exists
if psql -U $DB_USER -lqt | cut -d \| -f 1 | grep -qw $DB_NAME; then
    echo ""
    echo "Warning: Database '$DB_NAME' already exists!"
    read -p "Do you want to drop and recreate it? (yes/no): " RECREATE

    if [ "$RECREATE" = "yes" ]; then
        echo "Dropping existing database..."
        psql -U $DB_USER -c "DROP DATABASE $DB_NAME;"
        echo "✓ Database dropped"
    else
        echo "Skipping database creation"
        exit 0
    fi
fi

# Create database
echo ""
echo "Creating database '$DB_NAME'..."
psql -U $DB_USER -c "CREATE DATABASE $DB_NAME;"
echo "✓ Database created"

# Enable pgvector extension
echo ""
echo "Enabling pgvector extension..."
psql -U $DB_USER -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS vector;"
echo "✓ pgvector extension enabled"

# Apply schema
if [ -f "schema.sql" ]; then
    echo ""
    echo "Applying database schema..."
    psql -U $DB_USER -d $DB_NAME -f schema.sql
    echo "✓ Schema applied successfully"
else
    echo ""
    echo "Warning: schema.sql not found in current directory"
    echo "Please run this script from the /home/pi/hybridrag/image directory"
    exit 1
fi

# Verify setup
echo ""
echo "Verifying database setup..."
TABLE_COUNT=$(psql -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
echo "✓ Found $TABLE_COUNT tables in database"

echo ""
echo "========================================="
echo "✓ Database setup completed successfully!"
echo "========================================="
echo ""
echo "Database name: $DB_NAME"
echo "Tables created:"
psql -U $DB_USER -d $DB_NAME -c "\dt"
echo ""
echo "Next steps:"
echo "1. Install Python dependencies: pip install torch torchvision transformers ultralytics psycopg2-binary"
echo "2. Update system_config.json with db_name: 'image_vector'"
echo "3. Run your application!"
echo ""
