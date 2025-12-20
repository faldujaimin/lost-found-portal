# models.py
from datetime import datetime
# REMOVE THIS LINE: from flask_sqlalchemy import SQLAlchemy  <-- DELETE THIS!
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

# IMPORT THE `db` OBJECT THAT WAS ALREADY INITIALIZED IN app.py
# Assuming models.py is in the same directory as app.py
from app import db # <--- THIS IS THE CRITICAL CHANGE FOR models.py

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    registration_no = db.Column(db.String(20), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    role = db.Column(db.String(20), default='student') # 'student', 'admin', 'hod'

    items_reported = db.relationship(
        'Item',
        backref='reporter',
        lazy=True,
        # Using lambda is safer here because `Item` might not be fully defined yet when `User` is being processed
        foreign_keys=[lambda: Item.reported_by_id]
    )
    items_deleted = db.relationship(
        'Item',
        backref='deleter',
        lazy=True,
        # Using lambda is safer here because `Item` might not be fully defined yet when `User` is being processed
        foreign_keys=[lambda: Item.deleted_by_id]
    )

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def is_admin(self):
        return self.role == 'admin'

    def is_hod(self):
        return self.role == 'hod'

    def __repr__(self):
        return f"User('{self.registration_no}', '{self.full_name}', '{self.role}')"

class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(10), default='Lost', nullable=False) # 'Lost', 'Found', 'Reclaimed'
    reported_at = db.Column(db.DateTime, default=datetime.utcnow)
    lost_found_date = db.Column(db.Date, nullable=False)
    location = db.Column(db.String(200), nullable=False)
    image_filename = db.Column(db.String(255), nullable=True)

    is_active = db.Column(db.Boolean, default=True)

    reported_by_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    deleted_by_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    deleted_at = db.Column(db.DateTime, nullable=True)

    def __repr__(self):
        reporter_name = self.reporter.full_name if self.reporter else 'N/A'
        return f"<Item {self.item_name} - {self.status} by {reporter_name}>"