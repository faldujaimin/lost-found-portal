import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()

def test_connection():
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        print("❌ Error: DATABASE_URL not found in environment.")
        return

    # Fix for Render's postgres:// if needed (replicating logic from config.py)
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    print(f"Testing connection to: {db_url.split('@')[1] if '@' in db_url else 'Unknown'}") # Hide credentials

    try:
        engine = create_engine(db_url)
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("✅ Successfully connected to the database!")
            
            # Optional: Check if tables exist
            result = connection.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
            tables = [row[0] for row in result]
            if tables:
                print(f"✅ Found {len(tables)} tables: {', '.join(tables)}")
            else:
                print("⚠️  Connected, but no tables found (database might be empty).")

    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    test_connection()
