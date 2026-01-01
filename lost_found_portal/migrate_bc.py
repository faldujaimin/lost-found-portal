from app import app, db
from sqlalchemy import text

def migrate():
    with app.app_context():
        print("Starting migration using SQLAlchemy engine...")
        
        # 1. Add column to item table
        try:
            # We use text() to execute raw SQL for DDL
            db.session.execute(text("ALTER TABLE item ADD COLUMN blockchain_hash VARCHAR(64)"))
            db.session.commit()
            print("Added blockchain_hash column to item table.")
        except Exception as e:
            db.session.rollback()
            if "already exists" in str(e).lower() or "duplicate column" in str(e).lower():
                print("blockchain_hash column already exists.")
            else:
                print(f"Error adding column: {e}")
        
        # 2. Create blockchain_block table
        try:
            db.create_all()
            print("Ensured all tables (including blockchain_block) exist.")
        except Exception as e:
            print(f"Error creating tables: {e}")
        
        print("Migration complete.")

if __name__ == "__main__":
    migrate()
