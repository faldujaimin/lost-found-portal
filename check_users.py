
from app import app, db
from sqlalchemy import text

def check_users():
    with app.app_context():
        try:
            with db.engine.connect() as conn:
                result = conn.execute(text("SELECT id, registration_no, full_name, role FROM \"user\""))
                rows = result.fetchall()
                print(f"Total users in DB: {len(rows)}")
                for r in rows:
                    print(f"ID: {r[0]}, RegNo: {r[1]}, Name: {r[2]}, Role: {r[3]}")
        except Exception as e:
            print(f"Error checking users: {e}")

if __name__ == "__main__":
    check_users()
