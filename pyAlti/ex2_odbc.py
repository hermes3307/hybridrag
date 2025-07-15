"""
PyODBC Altibase Example - Complete Implementation

This example demonstrates how to use pyodbc to connect to Altibase database,
showcasing all the advanced features available in pyodbc that were missing
in the custom Altibase driver.

Prerequisites:
- pip install pyodbc
- Altibase ODBC driver installed and configured
- ODBC data source configured for Altibase
"""

import pyodbc
import datetime
import decimal
from typing import List, Dict, Any, Optional
import logging

# Configure logging to see pyodbc warnings and info
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AltibasePyODBCManager:
    """
    Comprehensive Altibase database manager using pyodbc
    Demonstrates all advanced pyodbc features
    """
    
    def __init__(self):
        self.connection = None
        self.setup_pyodbc_options()
    
    def setup_pyodbc_options(self):
        """Configure pyodbc module-level options"""
        # Enable connection pooling for better performance
        pyodbc.pooling = True
        
        # Set default timeout
        pyodbc.timeout = 30
        
        # Display pyodbc version and capabilities
        logger.info(f"PyODBC Version: {pyodbc.version}")
        logger.info(f"API Level: {pyodbc.apilevel}")
        logger.info(f"Thread Safety: {pyodbc.threadsafety}")
        logger.info(f"Parameter Style: {pyodbc.paramstyle}")
    
    def connect_to_altibase(self, 
                           server: str = "localhost",
                           port: int = 20300,
                           database: str = "mydb",
                           user: str = "sys",
                           password: str = "manager",
                           dsn_name: str = "Altibase ODBC Driver") -> pyodbc.Connection:
        """
        Connect to Altibase using your configured ODBC DSN
        """
        
        # Method 1: Use the configured DSN (recommended for your setup)
        logger.info(f"Attempting DSN connection using: {dsn_name}")
        try:
            # Use your configured DSN with credentials
            dsn_string = f"DSN={dsn_name};UID={user};PWD={password};"
            self.connection = pyodbc.connect(dsn_string)
            logger.info(f"Connected to Altibase successfully using DSN: {dsn_name}")
            
            # Configure connection attributes
            self.configure_connection()
            return self.connection
            
        except pyodbc.Error as e:
            logger.warning(f"DSN connection failed: {e}")
        
        # Method 2: Try direct driver connection as fallback
        driver_names = [
            "Altibase ODBC Driver",  # Your configured driver name
            "Altibase",
            "ALTIBASE HDB ODBC Driver",
            "Altibase 7.1 ODBC Driver"
        ]
        
        for driver in driver_names:
            connection_string = (
                f"DRIVER={{{driver}}};"
                f"SERVER={server};"
                f"PORT={port};"
                f"DATABASE={database};"
                f"UID={user};"
                f"PWD={password};"
                f"NLS_USE=US7ASCII;"  # Based on your config
                f"TIMEOUT=30;"
                f"AUTOCOMMIT=FALSE;"
            )
            
            try:
                logger.info(f"Attempting connection with driver: {driver}")
                self.connection = pyodbc.connect(connection_string)
                logger.info(f"Connected to Altibase successfully using driver: {driver}")
                
                # Configure connection attributes
                self.configure_connection()
                return self.connection
                
            except pyodbc.Error as e:
                logger.warning(f"Driver '{driver}' failed: {e}")
                continue
        
        # Method 3: List available drivers for debugging
        self.list_available_drivers()
        
        raise pyodbc.Error("Could not connect to Altibase with configured settings")
    
    def connect_with_dsn(self, dsn_name: str = "Altibase ODBC Driver", 
                         user: str = "sys", password: str = "manager"):
        """
        Connect using your pre-configured DSN
        """
        try:
            connection_string = f"DSN={dsn_name};UID={user};PWD={password};"
            self.connection = pyodbc.connect(connection_string)
            logger.info(f"Connected using DSN: {dsn_name}")
            self.configure_connection()
            return self.connection
        except pyodbc.Error as e:
            logger.error(f"DSN connection failed: {e}")
            raise
    
    def configure_connection(self):
        """Configure connection-level attributes and type converters"""
        if not self.connection:
            raise RuntimeError("No active connection")
        
        # Set connection attributes
        try:
            # Set autocommit mode
            self.connection.autocommit = False
            
            # Set connection timeout
            self.connection.timeout = 30
            
            # Get connection information
            logger.info("Connection Information:")
            logger.info(f"  Database: {self.get_connection_info(pyodbc.SQL_DATABASE_NAME)}")
            logger.info(f"  DBMS Name: {self.get_connection_info(pyodbc.SQL_DBMS_NAME)}")
            logger.info(f"  DBMS Version: {self.get_connection_info(pyodbc.SQL_DBMS_VER)}")
            logger.info(f"  Driver Name: {self.get_connection_info(pyodbc.SQL_DRIVER_NAME)}")
            logger.info(f"  Driver Version: {self.get_connection_info(pyodbc.SQL_DRIVER_VER)}")
            
        except pyodbc.Error as e:
            logger.warning(f"Could not configure connection attributes: {e}")
        
        # Setup custom type converters
        self.setup_type_converters()
    
    def get_connection_info(self, info_type: int) -> str:
        """Get connection information using getinfo()"""
        try:
            return self.connection.getinfo(info_type)
        except pyodbc.Error:
            return "Unknown"
    
    def setup_type_converters(self):
        """Setup custom output type converters"""
        
        # Custom decimal converter
        def convert_decimal(value):
            """Convert database decimal to Python Decimal with proper precision"""
            if value is None:
                return None
            return decimal.Decimal(str(value))
        
        # Custom date converter
        def convert_date(value):
            """Convert database date ensuring proper timezone handling"""
            if value is None:
                return None
            if isinstance(value, datetime.date):
                return value
            return datetime.datetime.strptime(str(value), '%Y-%m-%d').date()
        
        # Custom binary converter
        def convert_binary(value):
            """Convert binary data with proper encoding"""
            if value is None:
                return None
            return bytes(value)
        
        try:
            # Add type converters
            self.connection.add_output_converter(pyodbc.SQL_DECIMAL, convert_decimal)
            self.connection.add_output_converter(pyodbc.SQL_NUMERIC, convert_decimal)
            self.connection.add_output_converter(pyodbc.SQL_TYPE_DATE, convert_date)
            self.connection.add_output_converter(pyodbc.SQL_BINARY, convert_binary)
            self.connection.add_output_converter(pyodbc.SQL_VARBINARY, convert_binary)
            
            logger.info("Custom type converters configured")
            
        except pyodbc.Error as e:
            logger.warning(f"Could not setup type converters: {e}")
    
    def demonstrate_basic_operations(self):
        """Demonstrate basic database operations"""
        cursor = self.connection.cursor()
        
        try:
            # Basic query execution
            logger.info("\n=== Basic Operations ===")
            
            # Simple query
            cursor.execute("SELECT SYSDATE FROM DUAL")
            result = cursor.fetchone()
            logger.info(f"Current database time: {result[0]}")
            
            # Parameterized query with named parameters
            cursor.execute("""
                SELECT USER, DATABASE() as current_db, VERSION() as version
                FROM DUAL
            """)
            
            row = cursor.fetchone()
            if row:
                logger.info(f"Current user: {row[0]}")
                logger.info(f"Current database: {row[1] if row[1] else 'N/A'}")
                logger.info(f"Database version: {row[2] if row[2] else 'N/A'}")
            
            # Handle cursor messages (warnings/info)
            self.display_cursor_messages(cursor)
            
        except pyodbc.Error as e:
            logger.error(f"Basic operations failed: {e}")
        finally:
            cursor.close()
    
    def demonstrate_schema_introspection(self):
        """Demonstrate pyodbc's powerful schema introspection capabilities"""
        cursor = self.connection.cursor()
        
        try:
            logger.info("\n=== Schema Introspection ===")
            
            # List all tables
            logger.info("Available tables:")
            tables = cursor.tables(tableType='TABLE')
            table_count = 0
            for table in tables:
                table_count += 1
                logger.info(f"  {table.table_schem}.{table.table_name} ({table.table_type})")
                if table_count >= 10:  # Limit output
                    logger.info("  ... (showing first 10 tables)")
                    break
            
            # Get columns for a specific table (using system tables if available)
            try:
                cursor.execute("""
                    SELECT TABLE_NAME 
                    FROM SYSTEM_.SYS_TABLES_ 
                    WHERE USER_ID = (SELECT USER_ID FROM SYSTEM_.SYS_USERS_ WHERE USER_NAME = ?)
                    AND ROWNUM <= 5
                """, [self.get_current_user()])
                
                tables = cursor.fetchall()
                
                if tables:
                    table_name = tables[0][0]
                    logger.info(f"\nColumns for table {table_name}:")
                    
                    columns = cursor.columns(table=table_name)
                    for column in columns:
                        logger.info(f"  {column.column_name}: {column.type_name}"
                                  f"({column.column_size}) "
                                  f"{'NULL' if column.nullable else 'NOT NULL'}")
                        
            except pyodbc.Error as e:
                logger.warning(f"Could not retrieve table columns: {e}")
            
            # Get primary keys
            try:
                if 'table_name' in locals():
                    logger.info(f"\nPrimary keys for {table_name}:")
                    pks = cursor.primaryKeys(table=table_name)
                    for pk in pks:
                        logger.info(f"  {pk.column_name} (position: {pk.key_seq})")
            except pyodbc.Error as e:
                logger.warning(f"Could not retrieve primary keys: {e}")
            
            # Get data type information
            logger.info("\nSupported data types:")
            types = cursor.getTypeInfo()
            type_count = 0
            for type_info in types:
                type_count += 1
                logger.info(f"  {type_info.type_name}: SQL Type {type_info.data_type}")
                if type_count >= 10:  # Limit output
                    logger.info("  ... (showing first 10 types)")
                    break
                    
        except pyodbc.Error as e:
            logger.error(f"Schema introspection failed: {e}")
        finally:
            cursor.close()
    
    def list_available_drivers(self):
        """List all available ODBC drivers on the system"""
        logger.info("\n=== Available ODBC Drivers ===")
        try:
            drivers = pyodbc.drivers()
            logger.info("Installed ODBC drivers:")
            for i, driver in enumerate(drivers, 1):
                logger.info(f"  {i}. {driver}")
                
            if not drivers:
                logger.warning("No ODBC drivers found!")
            
            # Check for Altibase-related drivers
            altibase_drivers = [d for d in drivers if 'altibase' in d.lower() or 'hdb' in d.lower()]
            if altibase_drivers:
                logger.info(f"\nFound potential Altibase drivers: {altibase_drivers}")
            else:
                logger.warning("No Altibase ODBC drivers found!")
                self.show_installation_instructions()
                
        except Exception as e:
            logger.error(f"Could not list ODBC drivers: {e}")
    
    def show_installation_instructions(self):
        """Show instructions for installing Altibase ODBC driver"""
        logger.info("\n=== Altibase ODBC Driver Installation Instructions ===")
        logger.info("To use pyodbc with Altibase, you need to install the Altibase ODBC driver:")
        logger.info("1. Download Altibase ODBC driver from: http://support.altibase.com")
        logger.info("2. Install the driver for your platform:")
        logger.info("   - Windows: Run the .msi installer")
        logger.info("   - Linux: Install .rpm/.deb package or compile from source")
        logger.info("   - macOS: Install .pkg package")
        logger.info("3. Configure ODBC data source (optional):")
        logger.info("   - Windows: Use ODBC Data Source Administrator")
        logger.info("   - Linux/macOS: Edit /etc/odbc.ini or ~/.odbc.ini")
        logger.info("4. Verify installation by checking available drivers")
    
    def demonstrate_fallback_connection(self):
        """Demonstrate connection using mock data when Altibase is not available"""
        logger.info("\n=== Fallback Demo (Altibase not available) ===")
        logger.info("Since Altibase ODBC driver is not available, demonstrating with mock data...")
        
        # Create mock cursor and connection for demonstration
        class MockConnection:
            def __init__(self):
                self.autocommit = False
                self.timeout = 30
            
            def cursor(self):
                return MockCursor()
            
            def commit(self):
                logger.info("Mock: Transaction committed")
            
            def rollback(self):
                logger.info("Mock: Transaction rolled back")
            
            def close(self):
                logger.info("Mock: Connection closed")
        
        class MockCursor:
            def __init__(self):
                self.description = [
                    ('id', pyodbc.SQL_INTEGER, None, None, 10, 0, False),
                    ('name', pyodbc.SQL_VARCHAR, None, None, 100, 0, True),
                    ('email', pyodbc.SQL_VARCHAR, None, None, 200, 0, True),
                    ('created_date', pyodbc.SQL_TYPE_DATE, None, None, 10, 0, True),
                    ('salary', pyodbc.SQL_DECIMAL, None, None, 10, 2, True)
                ]
                self.rowcount = 5
                self.arraysize = 1
                self._data = [
                    (1, 'John Doe', 'john@example.com', datetime.date(2023, 1, 15), decimal.Decimal('55000.00')),
                    (2, 'Jane Smith', 'jane@example.com', datetime.date(2023, 2, 20), decimal.Decimal('62000.50')),
                    (3, 'Bob Johnson', 'bob@example.com', datetime.date(2023, 3, 10), decimal.Decimal('48000.75')),
                    (4, 'Alice Brown', 'alice@example.com', datetime.date(2023, 4, 5), decimal.Decimal('71000.25')),
                    (5, 'Charlie Wilson', 'charlie@example.com', datetime.date(2023, 5, 12), decimal.Decimal('58000.00'))
                ]
                self._index = 0
            
            def execute(self, sql, params=None):
                logger.info(f"Mock: Executing SQL: {sql[:50]}...")
                if params:
                    logger.info(f"Mock: Parameters: {params}")
                self._index = 0
            
            def fetchone(self):
                if self._index < len(self._data):
                    row = self._data[self._index]
                    self._index += 1
                    return row
                return None
            
            def fetchmany(self, size=1):
                rows = []
                for _ in range(size):
                    row = self.fetchone()
                    if row is None:
                        break
                    rows.append(row)
                return rows
            
            def fetchall(self):
                rows = self._data[self._index:]
                self._index = len(self._data)
                return rows
            
            def close(self):
                logger.info("Mock: Cursor closed")
        
        # Demonstrate with mock connection
        mock_conn = MockConnection()
        cursor = mock_conn.cursor()
        
        # Show what would happen with real Altibase connection
        logger.info("\nMock query execution:")
        cursor.execute("SELECT id, name, email, created_date, salary FROM employees WHERE salary > ?", [50000])
        
        logger.info("\nColumn descriptions:")
        for i, col in enumerate(cursor.description):
            logger.info(f"  {i}: {col[0]} (SQL type: {col[1]}) size: {col[4]} scale: {col[5]}")
        
        logger.info(f"\nRows affected: {cursor.rowcount}")
        
        logger.info("\nFetching results:")
        row = cursor.fetchone()
        if row:
            logger.info(f"First row: ID={row[0]}, Name={row[1]}, Email={row[2]}, Date={row[3]}, Salary=${row[4]}")
        
        remaining = cursor.fetchall()
        logger.info(f"Remaining {len(remaining)} rows:")
        for row in remaining:
            logger.info(f"  {row[0]}: {row[1]} - ${row[4]}")
        
        cursor.close()
        mock_conn.close()
        
        return mock_conn
    
    def demonstrate_advanced_querying(self):
        """Demonstrate advanced querying features"""
        cursor = self.connection.cursor()
        
        try:
            logger.info("\n=== Advanced Querying ===")
            
            # Create sample table for demonstration
            self.create_sample_table()
            
            # Bulk insert with executemany and fast_executemany
            logger.info("Performing bulk insert...")
            
            # Enable fast executemany for better performance
            cursor.fast_executemany = True
            
            sample_data = [
                (i, f"User_{i}", f"user{i}@example.com", 
                 datetime.date(2020 + (i % 4), 1 + (i % 12), 1 + (i % 28)),
                 decimal.Decimal(f"{1000 + i}.50"))
                for i in range(1, 101)
            ]
            
            cursor.executemany("""
                INSERT INTO demo_users (id, name, email, created_date, salary) 
                VALUES (?, ?, ?, ?, ?)
            """, sample_data)
            
            logger.info(f"Inserted {cursor.rowcount} rows")
            
            # Demonstrate different fetching methods
            cursor.execute("""
                SELECT id, name, email, created_date, salary 
                FROM demo_users 
                WHERE salary > ? 
                ORDER BY id
            """, [decimal.Decimal("1050.00")])
            
            # Get column information
            logger.info("\nQuery result columns:")
            for i, column in enumerate(cursor.description):
                logger.info(f"  {i}: {column[0]} ({column[1]}) "
                          f"size={column[3]} precision={column[4]} scale={column[5]}")
            
            # Fetch one row
            logger.info("\nFirst result:")
            first_row = cursor.fetchone()
            if first_row:
                logger.info(f"  ID: {first_row[0]}, Name: {first_row[1]}, "
                          f"Email: {first_row[2]}, Date: {first_row[3]}, "
                          f"Salary: {first_row[4]}")
            
            # Fetch multiple rows
            logger.info("\nNext 3 results:")
            next_rows = cursor.fetchmany(3)
            for row in next_rows:
                logger.info(f"  {row[0]}: {row[1]} - ${row[4]}")
            
            # Demonstrate cursor iteration
            logger.info("\nIterating through remaining results (first 5):")
            count = 0
            for row in cursor:
                count += 1
                logger.info(f"  {row[0]}: {row[1]} - ${row[4]}")
                if count >= 5:
                    break
            
            # Use fetchval() for single value queries
            cursor.execute("SELECT COUNT(*) FROM demo_users WHERE salary > ?", 
                          [decimal.Decimal("1070.00")])
            count = cursor.fetchval()
            logger.info(f"\nUsers with salary > $1070: {count}")
            
            # Demonstrate parameter binding with different types
            cursor.execute("""
                SELECT * FROM demo_users 
                WHERE created_date BETWEEN ? AND ? 
                AND name LIKE ?
                LIMIT 5
            """, [
                datetime.date(2022, 1, 1),
                datetime.date(2023, 12, 31),
                "User_1%"
            ])
            
            logger.info("\nFiltered results:")
            results = cursor.fetchall()
            for row in results:
                logger.info(f"  {row[1]} created on {row[3]}")
            
        except pyodbc.Error as e:
            logger.error(f"Advanced querying failed: {e}")
        finally:
            cursor.close()
    
    def create_sample_table(self):
        """Create a sample table for demonstrations"""
        cursor = self.connection.cursor()
        
        try:
            # Drop table if exists
            try:
                cursor.execute("DROP TABLE demo_users")
            except pyodbc.Error:
                pass  # Table doesn't exist
            
            # Create table
            cursor.execute("""
                CREATE TABLE demo_users (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    email VARCHAR(200),
                    created_date DATE,
                    salary DECIMAL(10,2)
                )
            """)
            
            self.connection.commit()
            logger.info("Sample table 'demo_users' created")
            
        except pyodbc.Error as e:
            logger.error(f"Failed to create sample table: {e}")
            raise
        finally:
            cursor.close()
    
    def demonstrate_transaction_handling(self):
        """Demonstrate advanced transaction handling"""
        logger.info("\n=== Transaction Handling ===")
        
        try:
            # Transaction with commit
            logger.info("Starting transaction...")
            cursor = self.connection.cursor()
            
            cursor.execute("INSERT INTO demo_users (id, name, email) VALUES (?, ?, ?)",
                          [9999, "Transaction Test", "test@example.com"])
            
            logger.info(f"Inserted row, rowcount: {cursor.rowcount}")
            
            # Check if row exists before commit
            cursor.execute("SELECT COUNT(*) FROM demo_users WHERE id = ?", [9999])
            count_before = cursor.fetchval()
            logger.info(f"Rows with ID 9999 before commit: {count_before}")
            
            self.connection.commit()
            logger.info("Transaction committed")
            
            # Transaction with rollback
            logger.info("Starting transaction for rollback...")
            cursor.execute("INSERT INTO demo_users (id, name, email) VALUES (?, ?, ?)",
                          [9998, "Rollback Test", "rollback@example.com"])
            
            logger.info("Inserted row (will be rolled back)")
            
            self.connection.rollback()
            logger.info("Transaction rolled back")
            
            # Verify rollback
            cursor.execute("SELECT COUNT(*) FROM demo_users WHERE id = ?", [9998])
            count_after = cursor.fetchval()
            logger.info(f"Rows with ID 9998 after rollback: {count_after}")
            
            cursor.close()
            
        except pyodbc.Error as e:
            logger.error(f"Transaction handling failed: {e}")
            self.connection.rollback()
    
    def demonstrate_error_handling(self):
        """Demonstrate comprehensive error handling"""
        logger.info("\n=== Error Handling ===")
        
        cursor = self.connection.cursor()
        
        # Demonstrate different types of errors
        error_tests = [
            ("Syntax Error", "SELECT * FORM invalid_table"),
            ("Table Not Found", "SELECT * FROM nonexistent_table"),
            ("Column Not Found", "SELECT invalid_column FROM demo_users"),
            ("Constraint Violation", "INSERT INTO demo_users (id, name) VALUES (1, 'Duplicate')")
        ]
        
        for test_name, sql in error_tests:
            try:
                logger.info(f"\nTesting {test_name}:")
                cursor.execute(sql)
                
            except pyodbc.InterfaceError as e:
                logger.error(f"  Interface Error: {e}")
            except pyodbc.DatabaseError as e:
                logger.error(f"  Database Error: {e}")
            except pyodbc.DataError as e:
                logger.error(f"  Data Error: {e}")
            except pyodbc.OperationalError as e:
                logger.error(f"  Operational Error: {e}")
            except pyodbc.IntegrityError as e:
                logger.error(f"  Integrity Error: {e}")
            except pyodbc.InternalError as e:
                logger.error(f"  Internal Error: {e}")
            except pyodbc.ProgrammingError as e:
                logger.error(f"  Programming Error: {e}")
            except pyodbc.NotSupportedError as e:
                logger.error(f"  Not Supported Error: {e}")
            except pyodbc.Error as e:
                logger.error(f"  General Error: {e}")
                # Display SQLSTATE if available
                if len(e.args) >= 2:
                    logger.error(f"  SQLSTATE: {e.args[0]}")
                    logger.error(f"  Message: {e.args[1]}")
        
        cursor.close()
    
    def display_cursor_messages(self, cursor):
        """Display cursor messages (warnings and informational messages)"""
        if hasattr(cursor, 'messages') and cursor.messages:
            logger.info("Cursor messages:")
            for message in cursor.messages:
                logger.info(f"  {message}")
    
    def demonstrate_performance_features(self):
        """Demonstrate performance optimization features"""
        logger.info("\n=== Performance Features ===")
        
        cursor = self.connection.cursor()
        
        try:
            # Set array size for fetching
            cursor.arraysize = 100
            logger.info(f"Set cursor arraysize to {cursor.arraysize}")
            
            # Demonstrate setinputsizes for parameter optimization
            cursor.setinputsizes([(pyodbc.SQL_INTEGER, 0, 0),
                                 (pyodbc.SQL_VARCHAR, 100, 0),
                                 (pyodbc.SQL_VARCHAR, 200, 0)])
            
            logger.info("Set input sizes for parameter optimization")
            
            # Demonstrate setoutputsize for large column optimization
            cursor.setoutputsize(1000, 1)  # Optimize column 1 for 1000 bytes
            logger.info("Set output size optimization")
            
            # Performance timing example
            import time
            
            start_time = time.time()
            cursor.execute("SELECT * FROM demo_users")
            fetch_time = time.time()
            
            all_rows = cursor.fetchall()
            end_time = time.time()
            
            logger.info(f"Query execution time: {fetch_time - start_time:.4f} seconds")
            logger.info(f"Fetch time: {end_time - fetch_time:.4f} seconds")
            logger.info(f"Total rows fetched: {len(all_rows)}")
            
        except pyodbc.Error as e:
            logger.error(f"Performance demonstration failed: {e}")
        finally:
            cursor.close()
    
    def cleanup(self):
        """Clean up resources"""
        if self.connection:
            try:
                # Clean up sample table
                cursor = self.connection.cursor()
                cursor.execute("DROP TABLE IF EXISTS demo_users")
                self.connection.commit()
                cursor.close()
                logger.info("Cleaned up sample table")
            except pyodbc.Error:
                pass  # Ignore cleanup errors
            
            # Close connection
            self.connection.close()
            logger.info("Connection closed")
    
    def get_current_user(self) -> str:
        """Get current database user"""
        cursor = self.connection.cursor()
        try:
            cursor.execute("SELECT USER FROM DUAL")
            result = cursor.fetchone()
            return result[0] if result else "UNKNOWN"
        except:
            return "SYS"  # Default fallback
        finally:
            cursor.close()
    
    def run_complete_demo(self):
        """Run the complete demonstration with your ODBC configuration"""
        try:
            # Connect using your configured DSN first
            logger.info("Attempting connection with your configured ODBC DSN...")
            self.connect_to_altibase(
                server="localhost",
                port=20300,
                database="mydb", 
                user="sys",
                password="manager",  # Update this with your actual password
                dsn_name="Altibase ODBC Driver"
            )
            
            # Run all demonstrations if connection successful
            self.demonstrate_basic_operations()
            self.demonstrate_schema_introspection()
            self.demonstrate_advanced_querying()
            self.demonstrate_transaction_handling()
            self.demonstrate_performance_features()
            self.demonstrate_error_handling()
            
        except pyodbc.Error as e:
            logger.error(f"Could not connect to Altibase: {e}")
            logger.info("This might be because:")
            logger.info("1. Altibase server is not running")
            logger.info("2. Wrong credentials (check your password)")
            logger.info("3. Network connectivity issues") 
            logger.info("4. Database 'mydb' doesn't exist")
            logger.info("\nRunning fallback demonstration...")
            
            # Show what pyodbc features would be available
            self.demonstrate_fallback_connection()
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
        finally:
            if hasattr(self, 'connection') and self.connection:
                self.cleanup()
            else:
                logger.info("No cleanup needed - connection was not established")

def main():
    """Main function to run the complete pyodbc Altibase demonstration"""
    print("PyODBC Altibase Comprehensive Example")
    print("=" * 50)
    
    # Create manager instance
    manager = AltibasePyODBCManager()
    
    # Use your configured ODBC settings
    print("Using your configured Altibase ODBC DSN...")
    print("DSN Name: Altibase ODBC Driver")
    print("Host: localhost")
    print("Port: 20300") 
    print("User: sys")
    print("Database: mydb")
    print("NLS_USE: US7ASCII")
    print("-" * 50)
    
    # Run complete demonstration
    manager.run_complete_demo()
    
    print("\nDemo completed! Check the logs above for detailed output.")

if __name__ == "__main__":
    main()