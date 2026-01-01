# One-off migration script to add avatar_filename column to user table if missing.
# Run with: python scripts/add_user_avatar_column.py
import sqlite3
import os
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_PATH = os.path.join(PROJECT_ROOT, 'instance', 'site.db')

if not os.path.exists(DB_PATH):
    print('Database not found at', DB_PATH)
    raise SystemExit(1)

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Query columns
c.execute("PRAGMA table_info(user);")
cols = [r[1] for r in c.fetchall()]
print('Existing columns:', cols)

if 'avatar_filename' in cols:
    print('avatar_filename already exists; nothing to do.')
else:
    print('Adding avatar_filename column to user table...')
    try:
        c.execute("ALTER TABLE user ADD COLUMN avatar_filename TEXT;")
        conn.commit()
        print('Column added successfully.')
    except Exception as e:
        print('Failed to add column:', e)

conn.close()