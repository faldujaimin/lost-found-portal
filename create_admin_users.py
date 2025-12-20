"""Utility script to create admin/HOD users for the lost_found_portal app.

Usage:
  - Run without arguments to (re)create default ADMIN001 and HOD001 users:
      python create_admin_users.py

  - Create a single user with arguments:
      python create_admin_users.py REGNO "Full Name" email role password

Note: Run this from the project folder where `app.py` is located.
"""

from app import app, db, User

def create_user(reg_no, full_name, email, role, password):
    with app.app_context():
        existing = User.query.filter_by(registration_no=reg_no).first()
        if existing:
            print(f"User with registration_no '{reg_no}' already exists (role={existing.role}). Skipping.")
            return

        u = User(registration_no=reg_no, full_name=full_name, email=email, role=role)
        u.set_password(password)
        db.session.add(u)
        db.session.commit()
        print(f"Created user: {reg_no} (role={role})")


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        # Create default admin and hod (same as app startup defaults)
        create_user('ADMIN001', 'Admin User', 'admin@example.com', 'admin', 'adminpassword')
        create_user('HOD001', 'HOD User', 'hod@example.com', 'hod', 'hodpassword')
        print('\nDone. You can now log in with those credentials on the /login page.')
    elif len(sys.argv) >= 6:
        # python create_admin_users.py REGNO "Full Name" email role password
        _, regno, fullname, email, role, password = sys.argv[:6]
        create_user(regno, fullname, email, role, password)
    else:
        print('Usage:')
        print('  python create_admin_users.py')
        print('  python create_admin_users.py REGNO "Full Name" email role password')