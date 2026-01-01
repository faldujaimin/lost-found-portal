
import sys
import os
from bs4 import BeautifulSoup

# Add current dir to path
sys.path.append(os.getcwd())

from app import app, db, User

def test_duplicate_register_verbose():
    reg_no = "TEST_DUP_002"
    email = "testdup002@example.com"
    
    with app.app_context():
        # Ensure user exists
        if not User.query.filter_by(registration_no=reg_no).first():
            print(f"Creating user {reg_no}...")
            u = User(registration_no=reg_no, full_name='T', email=email, role='student')
            u.set_password('pw')
            db.session.add(u)
            db.session.commit()
            
    print("Attempting duplicate registration...")
    with app.test_client() as client:
        response = client.post('/register', data={
            'registration_no': reg_no,
            'full_name': 'Test User Duplicate',
            'email': 'other2@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        }, follow_redirects=True)
        
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            soup = BeautifulSoup(response.data, 'html.parser')
            # Look for error messages
            errors = soup.find_all(class_='invalid-feedback')
            if errors:
                print("Validation Errors found:")
                for err in errors:
                    print(f"- {err.get_text(strip=True)}")
            
            alerts = soup.find_all(class_='alert')
            if alerts:
                 print("Alerts found:")
                 for alert in alerts:
                     print(f"- {alert.get_text(strip=True)}")
                     
            if not errors and not alerts:
                if "Your account has been created" in response.data.decode():
                     print("FAILURE: Duplicate Succeeded")
                else:
                     print("UNKNOWN STATE: No errors, no success message.")
                     # print(response.data.decode()[:500])
        else:
            print("FAILURE: Status not 200")
            print(response.data.decode())

if __name__ == "__main__":
    test_duplicate_register_verbose()
