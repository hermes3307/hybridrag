#!/bin/bash

# Database Management Helper Script
# Provides convenient commands for managing the PostgreSQL database

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

DB_NAME="${POSTGRES_DB:-vector_db}"
DB_USER="${POSTGRES_USER:-postgres}"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
Database Management Helper Script

Usage: ./db_manage.sh [command]

Commands:
  start       Start PostgreSQL service
  stop        Stop PostgreSQL service
  status      Check PostgreSQL service status
  restart     Restart PostgreSQL service

  connect     Connect to database with psql
  stats       Show database statistics
  backup      Create database backup
  restore     Restore database from backup

  reset       Reset database (WARNING: deletes all data)
  test        Run connection test

  help        Show this help message

Environment:
  Database: $DB_NAME
  User: $DB_USER

EOF
}

cmd_start() {
    log_info "Starting PostgreSQL service..."
    sudo service postgresql start
    sleep 1
    sudo service postgresql status
    log_success "PostgreSQL started"
}

cmd_stop() {
    log_info "Stopping PostgreSQL service..."
    sudo service postgresql stop
    log_success "PostgreSQL stopped"
}

cmd_status() {
    sudo service postgresql status
}

cmd_restart() {
    log_info "Restarting PostgreSQL service..."
    sudo service postgresql restart
    sleep 1
    sudo service postgresql status
    log_success "PostgreSQL restarted"
}

cmd_connect() {
    log_info "Connecting to database '$DB_NAME'..."
    echo "Available commands:"
    echo "  \\dt          - List tables"
    echo "  \\d faces     - Describe faces table"
    echo "  \\q           - Quit"
    echo ""
    sudo -u postgres psql -d "$DB_NAME"
}

cmd_stats() {
    log_info "Database statistics for '$DB_NAME'..."
    sudo -u postgres psql -d "$DB_NAME" -c "SELECT * FROM get_database_stats();"
    echo ""
    sudo -u postgres psql -d "$DB_NAME" -c "SELECT COUNT(*) as total_faces FROM faces;"
}

cmd_backup() {
    BACKUP_FILE="backup_${DB_NAME}_$(date +%Y%m%d_%H%M%S).sql"
    log_info "Creating backup of '$DB_NAME'..."
    sudo -u postgres pg_dump "$DB_NAME" > "$BACKUP_FILE"

    # Compress backup
    gzip "$BACKUP_FILE"
    log_success "Backup created: ${BACKUP_FILE}.gz"
}

cmd_restore() {
    echo "Available backups:"
    ls -lh backup_*.sql.gz 2>/dev/null || echo "No backups found"
    echo ""
    read -p "Enter backup file name (e.g., backup_vector_db_20251030_120000.sql.gz): " BACKUP_FILE

    if [ ! -f "$BACKUP_FILE" ]; then
        log_error "Backup file not found: $BACKUP_FILE"
        exit 1
    fi

    log_info "Restoring from $BACKUP_FILE..."

    # Decompress and restore
    gunzip -c "$BACKUP_FILE" | sudo -u postgres psql "$DB_NAME"
    log_success "Database restored"
}

cmd_reset() {
    log_error "WARNING: This will delete ALL data from the database!"
    read -p "Are you sure? Type 'yes' to confirm: " -r
    echo

    if [ "$REPLY" = "yes" ]; then
        log_info "Resetting database..."
        sudo -u postgres psql -d "$DB_NAME" -c "TRUNCATE TABLE faces RESTART IDENTITY;"
        log_success "Database reset complete"
    else
        log_info "Operation cancelled"
    fi
}

cmd_test() {
    log_info "Testing database connection..."

    if [ -f "test_pgvector.py" ]; then
        python3 test_pgvector.py
    else
        sudo -u postgres psql -d "$DB_NAME" -c "SELECT version();"
        sudo -u postgres psql -d "$DB_NAME" -c "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector';"
        log_success "Connection test passed"
    fi
}

# Main command dispatcher
case "${1:-help}" in
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    status)
        cmd_status
        ;;
    restart)
        cmd_restart
        ;;
    connect)
        cmd_connect
        ;;
    stats)
        cmd_stats
        ;;
    backup)
        cmd_backup
        ;;
    restore)
        cmd_restore
        ;;
    reset)
        cmd_reset
        ;;
    test)
        cmd_test
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
