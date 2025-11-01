╔════════════════════════════════════════════════════════════╗
║   Face Recognition System - Installation Quick Start      ║
╚════════════════════════════════════════════════════════════╝

AUTOMATED INSTALLATION (RECOMMENDED)
═══════════════════════════════════════════════════════════════

Run the automated installer:

    ./install.sh

This will:
  ✓ Install PostgreSQL and pgvector
  ✓ Create and configure the database
  ✓ Install Python dependencies
  ✓ Create configuration files
  ✓ Test the installation

═══════════════════════════════════════════════════════════════

WHAT'S INCLUDED
═══════════════════════════════════════════════════════════════

Scripts Created:
  • install.sh         - Automated installation
  • db_manage.sh       - Database management helper
  • .env.template      - Configuration template

Existing Scripts:
  • faces.py           - GUI application
  • search_cli.py      - Command-line search
  • search_examples.py - Interactive examples
  • test_pgvector.py   - Test suite
  • migrate_to_pgvector.py - Data migration tool

═══════════════════════════════════════════════════════════════

QUICK START GUIDE
═══════════════════════════════════════════════════════════════

1. Install Everything:
   ./install.sh

2. Start PostgreSQL:
   ./db_manage.sh start

3. Run the Application:
   python3 faces.py

4. Test the System:
   python3 test_pgvector.py

═══════════════════════════════════════════════════════════════

DATABASE MANAGEMENT
═══════════════════════════════════════════════════════════════

Service Control:
  ./db_manage.sh start     - Start PostgreSQL
  ./db_manage.sh stop      - Stop PostgreSQL
  ./db_manage.sh status    - Check status
  ./db_manage.sh restart   - Restart PostgreSQL

Database Operations:
  ./db_manage.sh connect   - Open database shell
  ./db_manage.sh stats     - Show statistics
  ./db_manage.sh backup    - Create backup
  ./db_manage.sh restore   - Restore from backup
  ./db_manage.sh reset     - Reset database
  ./db_manage.sh test      - Test connection

═══════════════════════════════════════════════════════════════

CONFIGURATION
═══════════════════════════════════════════════════════════════

Default Settings (in .env):
  • Database: vector_db
  • User: postgres
  • Password: postgres
  • Host: localhost
  • Port: 5432

To customize:
  1. Copy: cp .env.template .env
  2. Edit: nano .env

═══════════════════════════════════════════════════════════════

USAGE EXAMPLES
═══════════════════════════════════════════════════════════════

Add face images to the 'faces' directory, then:

GUI Mode:
  python3 faces.py

Command-Line Search:
  ./search_cli.py

Interactive Examples:
  ./search_examples.py

Database Inspection:
  python3 inspect_database.py

═══════════════════════════════════════════════════════════════

MIGRATION FROM CHROMADB
═══════════════════════════════════════════════════════════════

If you have existing ChromaDB data:

Preview migration:
  python3 migrate_to_pgvector.py --dry-run

Perform migration:
  python3 migrate_to_pgvector.py

═══════════════════════════════════════════════════════════════

TROUBLESHOOTING
═══════════════════════════════════════════════════════════════

Installation Issues:
  • See INSTALLATION.md for detailed guide
  • Check install.sh output for errors

Connection Issues:
  • Verify PostgreSQL is running:
    ./db_manage.sh status
  • Test connection:
    ./db_manage.sh test
  • Check .env configuration

Database Issues:
  • View logs:
    sudo tail -f /var/log/postgresql/*.log
  • Reset database:
    ./db_manage.sh reset

═══════════════════════════════════════════════════════════════

DOCUMENTATION
═══════════════════════════════════════════════════════════════

  INSTALLATION.md      - Complete installation guide
  PGVECTOR_README.md   - pgvector implementation details
  SEARCH_GUIDE.md      - Search functionality guide
  QUICK_START.txt      - Application usage guide

═══════════════════════════════════════════════════════════════

SYSTEM REQUIREMENTS
═══════════════════════════════════════════════════════════════

  • Linux (Ubuntu/Debian/WSL)
  • PostgreSQL 12+
  • Python 3.7+
  • 500MB disk space
  • 2GB RAM (recommended)

═══════════════════════════════════════════════════════════════

SUPPORT
═══════════════════════════════════════════════════════════════

For issues:
  1. Check INSTALLATION.md
  2. Run ./db_manage.sh test
  3. Check PostgreSQL logs
  4. Verify .env configuration

═══════════════════════════════════════════════════════════════

GET STARTED NOW
═══════════════════════════════════════════════════════════════

    ./install.sh

═══════════════════════════════════════════════════════════════
Last Updated: 2025-10-30
