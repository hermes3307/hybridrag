#!/bin/bash
# Fix PostgreSQL Authentication for Image Search System

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  PostgreSQL Authentication Fix${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

echo -e "${YELLOW}This script will help fix PostgreSQL authentication issues${NC}"
echo ""

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo -e "${RED}Error: PostgreSQL is not installed!${NC}"
    echo -e "${YELLOW}Install it with:${NC}"
    echo "  sudo apt update"
    echo "  sudo apt install postgresql postgresql-contrib"
    exit 1
fi

echo -e "${GREEN}✓${NC} PostgreSQL is installed"

# Check if PostgreSQL is running
if ! systemctl is-active --quiet postgresql; then
    echo -e "${YELLOW}PostgreSQL is not running. Starting it...${NC}"
    sudo systemctl start postgresql
    echo -e "${GREEN}✓${NC} PostgreSQL started"
else
    echo -e "${GREEN}✓${NC} PostgreSQL is running"
fi

echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  Choose Authentication Method${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo "1) Use current user '$(whoami)' as database superuser (Recommended)"
echo "2) Set password for 'postgres' user and configure md5 authentication"
echo "3) Create database as current user only"
echo ""
read -p "Enter option [1-3]: " auth_choice
echo ""

case $auth_choice in
    1)
        # Option 1: Use current user
        echo -e "${YELLOW}Creating PostgreSQL superuser for current user...${NC}"

        # Create user if not exists
        sudo -u postgres psql -c "SELECT 1 FROM pg_user WHERE usename = '$(whoami)';" | grep -q 1 || \
        sudo -u postgres createuser -s $(whoami)

        echo -e "${GREEN}✓${NC} User '$(whoami)' is now a PostgreSQL superuser"

        # Create database
        echo -e "${YELLOW}Creating database 'image_vector'...${NC}"

        # Check if database exists
        if psql -lqt | cut -d \| -f 1 | grep -qw image_vector; then
            echo -e "${YELLOW}Database 'image_vector' already exists${NC}"
            read -p "Recreate it? (yes/no): " recreate
            if [ "$recreate" = "yes" ]; then
                dropdb image_vector
                echo -e "${GREEN}✓${NC} Dropped existing database"
            else
                echo -e "${YELLOW}Using existing database${NC}"
            fi
        fi

        if ! psql -lqt | cut -d \| -f 1 | grep -qw image_vector; then
            createdb image_vector
            echo -e "${GREEN}✓${NC} Database created"
        fi

        # Enable pgvector
        echo -e "${YELLOW}Enabling pgvector extension...${NC}"
        psql -d image_vector -c "CREATE EXTENSION IF NOT EXISTS vector;"
        echo -e "${GREEN}✓${NC} pgvector enabled"

        # Apply schema
        if [ -f "schema.sql" ]; then
            echo -e "${YELLOW}Applying schema...${NC}"
            psql -d image_vector -f schema.sql
            echo -e "${GREEN}✓${NC} Schema applied"
        fi

        # Update config files to use current user
        echo -e "${YELLOW}Updating configuration files...${NC}"

        # Update core.py
        if [ -f "core.py" ]; then
            sed -i "s/db_user: str = \"postgres\"/db_user: str = \"$(whoami)\"/" core.py
            sed -i "s/db_password: str = \"postgres\"/db_password: str = \"\"/" core.py
            echo -e "${GREEN}✓${NC} Updated core.py"
        fi

        # Update or create system_config.json
        cat > system_config.json << EOF
{
  "images_dir": "./images",
  "db_host": "localhost",
  "db_port": 5432,
  "db_name": "image_vector",
  "db_user": "$(whoami)",
  "db_password": "",
  "download_delay": 1.0,
  "max_workers": 2,
  "batch_size": 50,
  "embedding_model": "statistical",
  "download_source": "picsum_landscape",
  "config_file": "system_config.json"
}
EOF
        echo -e "${GREEN}✓${NC} Created system_config.json"

        echo ""
        echo -e "${GREEN}=========================================${NC}"
        echo -e "${GREEN}  Setup Complete!${NC}"
        echo -e "${GREEN}=========================================${NC}"
        echo ""
        echo -e "${GREEN}✓${NC} Database user: $(whoami)"
        echo -e "${GREEN}✓${NC} Database name: image_vector"
        echo -e "${GREEN}✓${NC} Authentication: peer (automatic)"
        echo ""
        echo -e "${YELLOW}Test the connection:${NC}"
        echo "  psql -d image_vector -c 'SELECT 1;'"
        ;;

    2)
        # Option 2: Set postgres password
        echo -e "${YELLOW}Setting password for 'postgres' user...${NC}"
        echo ""
        read -sp "Enter new password for postgres user: " POSTGRES_PASSWORD
        echo ""
        read -sp "Confirm password: " POSTGRES_PASSWORD2
        echo ""

        if [ "$POSTGRES_PASSWORD" != "$POSTGRES_PASSWORD2" ]; then
            echo -e "${RED}Passwords don't match!${NC}"
            exit 1
        fi

        # Set password
        sudo -u postgres psql -c "ALTER USER postgres PASSWORD '$POSTGRES_PASSWORD';"
        echo -e "${GREEN}✓${NC} Password set"

        # Update pg_hba.conf
        echo -e "${YELLOW}Updating pg_hba.conf for md5 authentication...${NC}"

        PG_HBA=$(sudo -u postgres psql -t -c "SHOW hba_file;" | xargs)
        echo -e "${YELLOW}pg_hba.conf location: $PG_HBA${NC}"

        # Backup
        sudo cp "$PG_HBA" "${PG_HBA}.backup.$(date +%Y%m%d_%H%M%S)"
        echo -e "${GREEN}✓${NC} Backup created"

        # Update local connections to use md5
        sudo sed -i 's/^local.*all.*postgres.*peer/local   all             postgres                                md5/' "$PG_HBA"
        sudo sed -i 's/^local.*all.*all.*peer/local   all             all                                     md5/' "$PG_HBA"

        echo -e "${GREEN}✓${NC} pg_hba.conf updated"

        # Reload PostgreSQL
        echo -e "${YELLOW}Reloading PostgreSQL...${NC}"
        sudo systemctl reload postgresql
        echo -e "${GREEN}✓${NC} PostgreSQL reloaded"

        # Update system_config.json
        cat > system_config.json << EOF
{
  "images_dir": "./images",
  "db_host": "localhost",
  "db_port": 5432,
  "db_name": "image_vector",
  "db_user": "postgres",
  "db_password": "$POSTGRES_PASSWORD",
  "download_delay": 1.0,
  "max_workers": 2,
  "batch_size": 50,
  "embedding_model": "statistical",
  "download_source": "picsum_landscape",
  "config_file": "system_config.json"
}
EOF
        echo -e "${GREEN}✓${NC} Created system_config.json with password"

        echo ""
        echo -e "${GREEN}=========================================${NC}"
        echo -e "${GREEN}  Setup Complete!${NC}"
        echo -e "${GREEN}=========================================${NC}"
        echo ""
        echo -e "${YELLOW}Now run the database setup:${NC}"
        echo "  ./setup_database.sh"
        ;;

    3)
        # Option 3: Simple user-only setup
        echo -e "${YELLOW}Creating database as user '$(whoami)'...${NC}"

        sudo -u postgres psql -c "SELECT 1 FROM pg_user WHERE usename = '$(whoami)';" | grep -q 1 || \
        sudo -u postgres createuser $(whoami)

        sudo -u postgres createdb -O $(whoami) image_vector 2>/dev/null || echo -e "${YELLOW}Database may already exist${NC}"

        psql -d image_vector -c "CREATE EXTENSION IF NOT EXISTS vector;"

        if [ -f "schema.sql" ]; then
            psql -d image_vector -f schema.sql
        fi

        cat > system_config.json << EOF
{
  "images_dir": "./images",
  "db_host": "localhost",
  "db_port": 5432,
  "db_name": "image_vector",
  "db_user": "$(whoami)",
  "db_password": "",
  "download_delay": 1.0,
  "max_workers": 2,
  "batch_size": 50,
  "embedding_model": "statistical",
  "download_source": "picsum_landscape",
  "config_file": "system_config.json"
}
EOF

        echo -e "${GREEN}✓${NC} Setup complete!"
        ;;

    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  Test Your Connection${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo "Run this command to test:"
echo "  psql -d image_vector -c 'SELECT version();'"
echo ""
