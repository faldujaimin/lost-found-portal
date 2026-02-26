
import sqlite3
import os

# Configuration
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'site.db')

def migrate_db():
    print(f"Checking database at: {DB_PATH}")
    if not os.path.exists(DB_PATH):
        print("Database file not found. Run the app to create a fresh database.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Add new columns to User table
    print("\n--- Migrating 'user' table ---")
    new_columns = [
        ('contact_visible', 'BOOLEAN DEFAULT 0 NOT NULL'),
        ('email_public', 'BOOLEAN DEFAULT 0 NOT NULL'),
        ('phone', 'VARCHAR(15)'),
        ('phone_public', 'BOOLEAN DEFAULT 0 NOT NULL'),
        ('preferred_language', "VARCHAR(5) DEFAULT 'en' NOT NULL"),
        ('created_at', "DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL"),
        ('last_login', 'DATETIME'),
        ('data_consent', 'BOOLEAN DEFAULT 1 NOT NULL')
    ]
    
    for col_name, col_def in new_columns:
        try:
            cursor.execute(f"ALTER TABLE user ADD COLUMN {col_name} {col_def}")
            print(f"Added column: {col_name}")
        except sqlite3.OperationalError as e:
            if 'duplicate column' in str(e).lower():
                print(f"Column already exists: {col_name}")
            else:
                print(f"Error adding {col_name}: {e}")

    # 2. Add Analytics table
    print("\n--- Checking 'analytics' table ---")
    try:
        cursor.execute("SELECT id FROM analytics LIMIT 1")
        print("'analytics' table already exists.")
    except sqlite3.OperationalError:
        print("Creating 'analytics' table...")
        cursor.execute('''
            CREATE TABLE analytics (
                id INTEGER PRIMARY KEY,
                date DATE NOT NULL UNIQUE,
                items_lost_count INTEGER DEFAULT 0,
                items_found_count INTEGER DEFAULT 0,
                items_returned_count INTEGER DEFAULT 0,
                new_users_count INTEGER DEFAULT 0,
                active_users_count INTEGER DEFAULT 0,
                total_logins INTEGER DEFAULT 0,
                success_rate FLOAT DEFAULT 0.0,
                avg_return_time_days FLOAT DEFAULT 0.0,
                popular_locations TEXT,
                popular_items TEXT,
                peak_hours TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("'analytics' table created successfully.")

    conn.commit()
    conn.close()
    print("\nMigration complete due to schema updates.")

if __name__ == "__main__":
    migrate_db()
