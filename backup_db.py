"""Simple SQLite backup utility for the Lost & Found Portal.

Usage:
    python backup_db.py         # creates a timestamped copy in instance/backups/

You can schedule this script with Windows Task Scheduler to run periodically.
"""
import os
import sqlite3
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
INSTANCE_DIR = os.path.join(PROJECT_ROOT, 'instance')
DB_FILENAME = 'site.db'
DB_PATH = os.path.join(INSTANCE_DIR, DB_FILENAME)
BACKUP_DIR = os.path.join(INSTANCE_DIR, 'backups')


def ensure_backup_dir():
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR, exist_ok=True)


def backup_sqlite(src_path, dest_path):
    """Use sqlite3's backup API to copy a live database safely."""
    # Connect to the source database in read-only mode to avoid interfering with writer
    src = sqlite3.connect(f'file:{src_path}?mode=ro', uri=True)
    dest = sqlite3.connect(dest_path)
    with dest:
        src.backup(dest, pages=0)
    src.close()
    dest.close()


if __name__ == '__main__':
    if not os.path.exists(DB_PATH):
        print(f"Database file not found at {DB_PATH}. Nothing to back up.")
        raise SystemExit(1)

    ensure_backup_dir()

    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    backup_name = f"site_{timestamp}.db"
    backup_path = os.path.join(BACKUP_DIR, backup_name)

    try:
        backup_sqlite(DB_PATH, backup_path)
        print(f"Backup created: {backup_path}")
    except Exception as e:
        print(f"Failed to create backup: {e}")
        raise SystemExit(1)
