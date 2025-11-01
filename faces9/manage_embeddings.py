#!/usr/bin/env python3
"""
Embedding Model Manager

Manage embeddings in the database by model type:
- List all embedding models and counts
- Delete embeddings by model
- Show statistics
"""

import sys
import argparse
import psycopg2

# Simple table formatter (no external dependencies)
def simple_table(data, headers):
    """Simple table formatter"""
    if not data:
        return ""

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Create separator
    separator = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"

    # Format header
    header_row = "| " + " | ".join([str(headers[i]).ljust(col_widths[i]) for i in range(len(headers))]) + " |"

    # Format rows
    rows = []
    for row in data:
        rows.append("| " + " | ".join([str(row[i]).ljust(col_widths[i]) for i in range(len(row))]) + " |")

    return "\n".join([separator, header_row, separator] + rows + [separator])

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'vector_db',
    'user': 'postgres',
    'password': 'postgres'
}


def get_db_connection():
    """Get database connection"""
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        sys.exit(1)


def list_models():
    """List all embedding models and their counts"""
    conn = get_db_connection()
    cur = conn.cursor()

    # Get counts by model
    cur.execute("""
        SELECT
            embedding_model,
            COUNT(*) as count,
            MIN(timestamp) as first_added,
            MAX(timestamp) as last_added
        FROM faces
        GROUP BY embedding_model
        ORDER BY count DESC
    """)

    results = cur.fetchall()
    conn.close()

    print("=" * 80)
    print("📊 EMBEDDING MODELS IN DATABASE")
    print("=" * 80)
    print()

    if not results:
        print("No embeddings found in database.")
        return

    headers = ["Model", "Count", "First Added", "Last Added"]
    table_data = []

    for model, count, first, last in results:
        table_data.append([
            model,
            f"{count:,}",
            first.strftime("%Y-%m-%d %H:%M") if first else "N/A",
            last.strftime("%Y-%m-%d %H:%M") if last else "N/A"
        ])

    print(simple_table(table_data, headers))
    print()

    total = sum(row[1] for row in results)
    print(f"Total embeddings: {total:,}")
    print()


def show_model_samples(model_name: str, limit: int = 5):
    """Show sample records for a model"""
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT face_id, timestamp
        FROM faces
        WHERE embedding_model = %s
        ORDER BY timestamp DESC
        LIMIT %s
    """, (model_name, limit))

    results = cur.fetchall()
    conn.close()

    if not results:
        print(f"No records found for model: {model_name}")
        return

    print(f"\nSample records (showing {len(results)}):")
    headers = ["Face ID", "Timestamp"]
    table_data = [[face_id, ts.strftime("%Y-%m-%d %H:%M:%S")] for face_id, ts in results]
    print(simple_table(table_data, headers))


def delete_by_model(model_name: str, dry_run: bool = True):
    """Delete embeddings by model"""
    conn = get_db_connection()
    cur = conn.cursor()

    # Get count
    cur.execute("SELECT COUNT(*) FROM faces WHERE embedding_model = %s", (model_name,))
    count = cur.fetchone()[0]

    if count == 0:
        print(f"✅ No embeddings found with model: {model_name}")
        conn.close()
        return

    print("=" * 80)
    print(f"🗑️  DELETE EMBEDDINGS - Model: {model_name}")
    print("=" * 80)
    print()
    print(f"Records to delete: {count:,}")
    print()

    # Show samples
    show_model_samples(model_name, 5)
    print()

    if dry_run:
        print("=" * 80)
        print("⚠️  DRY-RUN MODE - NO RECORDS DELETED")
        print("=" * 80)
        print()
        print("This was a dry-run. No records were actually deleted.")
        print()
        print("To actually delete these records, run:")
        print(f"  python3 manage_embeddings.py delete {model_name} --confirm")
        print()
    else:
        print("=" * 80)
        print("⚠️  DELETION MODE")
        print("=" * 80)
        print()
        print(f"⚠️  WARNING: You are about to delete {count:,} embeddings!")
        print(f"⚠️  Model: {model_name}")
        print("⚠️  This action CANNOT be undone!")
        print()

        response = input("Type 'DELETE' to confirm: ").strip()

        if response != "DELETE":
            print("❌ Deletion cancelled.")
            conn.close()
            return

        print()
        print("🗑️  Deleting records...")

        try:
            cur.execute("DELETE FROM faces WHERE embedding_model = %s", (model_name,))
            conn.commit()
            print(f"✅ Successfully deleted {count:,} records")
            print()

            # Show updated stats
            print("Updated database status:")
            list_models()

        except Exception as e:
            conn.rollback()
            print(f"❌ Error during deletion: {e}")
            conn.close()
            sys.exit(1)

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description='Manage embeddings in the database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all models
  python3 manage_embeddings.py list

  # Delete statistical embeddings (dry-run)
  python3 manage_embeddings.py delete statistical

  # Actually delete statistical embeddings
  python3 manage_embeddings.py delete statistical --confirm

  # Delete facenet embeddings
  python3 manage_embeddings.py delete facenet --confirm
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # List command
    subparsers.add_parser('list', help='List all embedding models')

    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete embeddings by model')
    delete_parser.add_argument('model', help='Model name to delete')
    delete_parser.add_argument('--confirm', action='store_true', help='Actually delete (not dry-run)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'list':
        list_models()
    elif args.command == 'delete':
        delete_by_model(args.model, dry_run=not args.confirm)


if __name__ == '__main__':
    main()
