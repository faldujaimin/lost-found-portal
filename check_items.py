
from app import app, db

def check_db():
    with app.app_context():
        # Reflect or use the classes already registered in db.Model
        # Since they are in app.py, they should be available in the engine/metadata
        from sqlalchemy import text
        
        # More robust way: query the DB directly to see what's in there
        try:
            with db.engine.connect() as conn:
                result = conn.execute(text("SELECT id, item_name, status, is_active FROM item"))
                rows = result.fetchall()
                print(f"Total items in DB: {len(rows)}")
                for r in rows:
                    print(f"ID: {r[0]}, Name: {r[1]}, Status: {r[2]}, Active: {r[3]}")
                
                print("\nRecent Logs:")
                result = conn.execute(text("SELECT timestamp, action, details FROM report_log ORDER BY timestamp DESC LIMIT 10"))
                for r in result:
                    print(f"TS: {r[0]}, Action: {r[1]}, Details: {r[2]}")
        except Exception as e:
            print(f"Error checking DB: {e}")

if __name__ == "__main__":
    check_db()
