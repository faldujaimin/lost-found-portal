import os
import sqlite3

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, 'instance', 'site.db')

if not os.path.exists(DB_PATH):
    print(f"Database not found at {DB_PATH}")
    raise SystemExit(1)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("PRAGMA table_info('user')")
cols = [r[1] for r in cur.fetchall()]
print('Existing columns:', cols)

if 'points' not in cols:
    print('Adding column: points')
    cur.execute("ALTER TABLE user ADD COLUMN points INTEGER DEFAULT 0")
else:
    print('Column points already present')

if 'level' not in cols:
    print('Adding column: level')
    cur.execute("ALTER TABLE user ADD COLUMN level TEXT DEFAULT 'Bronze'")
else:
    print('Column level already present')

conn.commit()
cur.execute("PRAGMA table_info('user')")
cols_after = [r[1] for r in cur.fetchall()]
print('Columns after migration:', cols_after)
conn.close()
print('Migration complete')
