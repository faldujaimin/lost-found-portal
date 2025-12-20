from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TextAreaField, DateField, FileField
from wtforms.validators import DataRequired, Length, EqualTo, ValidationError, Regexp, Optional
from wtforms.widgets import DateInput
from flask_wtf.file import FileAllowed # For file validation

# Assuming you have a User model in models.py
# from .models import User # If forms.py is in the same directory as models.py

class RegistrationForm(FlaskForm):
    registration_no = StringField('Registration No.', validators=[
        DataRequired(),
        Length(min=5, max=20),
        Regexp('^[A-Za-z0-9]+$', message="Registration number must contain only letters and digits.")
    ])
    full_name = StringField('Full Name', validators=[DataRequired(), Length(min=2, max=100)])
    email = StringField('Email (Optional)', validators=[Optional(), Length(max=120)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6, message="Password must be at least 6 characters long.")])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password', message="Passwords must match.")])
    submit = SubmitField('Register')

    # Custom validator for unique registration number
    def validate_registration_no(self, registration_no):
        # This will require importing User model in app.py if forms are separate.
        # Or if models.py is imported here, then from .models import User
        # For simplicity, if forms.py is separate, validation can be moved to route logic in app.py
        # Or, pass the User model to the form constructor.
        # A simpler way is to catch IntegrityError in the route itself.
        pass # The unique check is done in the route in app.py already

class LoginForm(FlaskForm):
    registration_no = StringField('Registration No.', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class ReportLostItemForm(FlaskForm):
    item_name = StringField('Item Name', validators=[DataRequired(), Length(max=100)])
    description = TextAreaField('Description', validators=[DataRequired()])
    lost_date = DateField('Date Lost', validators=[DataRequired()], widget=DateInput())
    location = StringField('Location Lost', validators=[DataRequired(), Length(max=200)])
    submit = SubmitField('Report Lost Item')

class ReportFoundItemForm(FlaskForm):
    item_name = StringField('Item Name', validators=[DataRequired(), Length(max=100)])
    description = TextAreaField('Description', validators=[DataRequired()])
    found_date = DateField('Date Found', validators=[DataRequired()], widget=DateInput())
    location = StringField('Location Found', validators=[DataRequired(), Length(max=200)])
    image = FileField('Upload Image', validators=[
        FileAllowed(['png', 'jpg', 'jpeg', 'gif'], 'Images only!'), # This checks file extension on server side
        DataRequired(message="An image is required for found items.")
    ])
    submit = SubmitField('Report Found Item')