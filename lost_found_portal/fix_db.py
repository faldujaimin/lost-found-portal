
import sqlite3
import os

# Define database path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'instance', 'site.db')

def fix_database():
    print(f"Target Database: {DB_PATH}")
    
    if not os.path.exists(DB_PATH):
        print("Error: Database file not found at expected path.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # List of new columns to ensure exist in 'user' table
    # Removed NOT NULL from created_at to avoid SQLite limitation with dynamic defaults
    new_columns = [
        ('contact_visible', 'BOOLEAN DEFAULT 0 NOT NULL'),
        ('email_public', 'BOOLEAN DEFAULT 0 NOT NULL'),
        ('phone', 'VARCHAR(15)'),
        ('phone_public', 'BOOLEAN DEFAULT 0 NOT NULL'),
        ('preferred_language', "VARCHAR(5) DEFAULT 'en' NOT NULL"),
        ('created_at', "DATETIME DEFAULT CURRENT_TIMESTAMP"), 
        ('last_login', 'DATETIME'),
        ('data_consent', 'BOOLEAN DEFAULT 1 NOT NULL')
    ]
    
    print("\nChecking 'user' table columns...")
    cursor.execute("PRAGMA table_info(user)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    print(f"Existing columns: {existing_cols}")
    
    for col_name, col_def in new_columns:
        if col_name not in existing_cols:
            try:
                print(f"Adding missing column: {col_name}...")
                cursor.execute(f"ALTER TABLE user ADD COLUMN {col_name} {col_def}")
                print(f" -> Success.")
            except Exception as e:
                print(f" -> Failed to add {col_name}: {e}")
                # Fallback for created_at if default fails
                if col_name == 'created_at':
                    try:
                        print(f" -> Retrying {col_name} as basic DATETIME...")
                        cursor.execute(f"ALTER TABLE user ADD COLUMN {col_name} DATETIME")
                        print(f" -> Success (Fallback).")
                    except Exception as e2:
                        print(f" -> Fallback failed: {e2}")

        else:
            print(f"Column '{col_name}' already exists.")

    conn.commit()
    conn.close()
    print("\nDatabase repair complete.")

if __name__ == "__main__":
    fix_database()
