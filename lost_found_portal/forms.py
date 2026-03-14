from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TextAreaField, DateField, FileField
from wtforms.validators import DataRequired, Length, EqualTo, ValidationError, Regexp, Optional
from wtforms.widgets import DateInput
from flask_wtf.file import FileAllowed
try:
    from flask_babel import lazy_gettext as _l
except ImportError:
    def _l(s): return s

# Assuming you have a User model in models.py
# from .models import User # If forms.py is in the same directory as models.py

class RegistrationForm(FlaskForm):
    registration_no = StringField(_l('Registration No.'), validators=[
        DataRequired(),
        Length(min=5, max=20),
        Regexp('^[A-Za-z0-9]+$', message=_l("Registration number must contain only letters and digits."))
    ])
    full_name = StringField(_l('Full Name'), validators=[DataRequired(), Length(min=2, max=100)])
    email = StringField(_l('Email (Optional)'), validators=[Optional(), Length(max=120)])
    password = PasswordField(_l('Password'), validators=[DataRequired(), Length(min=6, message=_l("Password must be at least 6 characters long."))])
    confirm_password = PasswordField(_l('Confirm Password'), validators=[DataRequired(), EqualTo('password', message=_l("Passwords must match."))])
    submit = SubmitField(_l('Register'))

    # Custom validator for unique registration number
    def validate_registration_no(self, registration_no):
        # This will require importing User model in app.py if forms are separate.
        # Or if models.py is imported here, then from .models import User
        # For simplicity, if forms.py is separate, validation can be moved to route logic in app.py
        # Or, pass the User model to the form constructor.
        # A simpler way is to catch IntegrityError in the route itself.
        pass # The unique check is done in the route in app.py already

class LoginForm(FlaskForm):
    registration_no = StringField(_l('Registration No.'), validators=[DataRequired()])
    password = PasswordField(_l('Password'), validators=[DataRequired()])
    submit = SubmitField(_l('Login'))

class ReportLostItemForm(FlaskForm):
    item_name = StringField(_l('Item Name'), validators=[DataRequired(), Length(max=100)])
    description = TextAreaField(_l('Description'), validators=[DataRequired()])
    lost_date = DateField(_l('Date Lost'), validators=[DataRequired()], widget=DateInput())
    location = StringField(_l('Location Lost'), validators=[DataRequired(), Length(max=200)])
    submit = SubmitField(_l('Report Lost Item'))

class ReportFoundItemForm(FlaskForm):
    item_name = StringField(_l('Item Name'), validators=[DataRequired(), Length(max=100)])
    description = TextAreaField(_l('Description'), validators=[DataRequired()])
    found_date = DateField(_l('Date Found'), validators=[DataRequired()], widget=DateInput())
    location = StringField(_l('Location Found'), validators=[DataRequired(), Length(max=200)])
    image = FileField(_l('Upload Image'), validators=[
        FileAllowed(['png', 'jpg', 'jpeg', 'gif'], _l('Images only!')), # This checks file extension on server side
        DataRequired(message=_l("An image is required for found items."))
    ])
    submit = SubmitField(_l('Report Found Item'))