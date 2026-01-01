"""
Database Migration Script for OCR and Verification Features
Adds new columns to the Item table for storing OCR and verification data
"""

from app import app, db
from sqlalchemy import text

def migrate_database():
    """Add new columns to Item table if they don't exist"""
    
    with app.app_context():
        print("Starting database migration...")
        
        # Check if columns already exist
        inspector = db.inspect(db.engine)
        existing_columns = [col['name'] for col in inspector.get_columns('item')]
        
        columns_to_add = {
            'ocr_extracted_text': 'TEXT',
            'verification_score': 'FLOAT',
            'verification_details': 'TEXT'
        }
        
        for col_name, col_type in columns_to_add.items():
            if col_name not in existing_columns:
                try:
                    with db.engine.connect() as conn:
                        conn.execute(text(f'ALTER TABLE item ADD COLUMN {col_name} {col_type}'))
                        conn.commit()
                    print(f"✓ Added column: {col_name}")
                except Exception as e:
                    print(f"✗ Failed to add column {col_name}: {e}")
            else:
                print(f"ℹ Column {col_name} already exists")
        
        print("\nDatabase migration completed successfully!")
        print("\nNew features available:")
        print("  - OCR text extraction from uploaded images")
        print("  - Cross-verification with confidence scoring")
        print("  - Enhanced UI with verification displays")

if __name__ == '__main__':
    migrate_database()
