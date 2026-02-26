import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a_very_secret_key_that_should_be_changed'
    # Default to PostgreSQL now that it is installed
    # database_url = os.environ.get('DATABASE_URL')
    # if database_url and database_url.startswith("postgres://"):
    #     database_url = database_url.replace("postgres://", "postgresql://", 1)
    
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'instance', 'site.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'static', 'uploads')
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2 Megabytes
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    # Multi-language support (i18n)
    BABEL_DEFAULT_LOCALE = 'en'
    BABEL_SUPPORTED_LOCALES = ['en', 'hi', 'gu']  # English, Hindi, Gujarati
    BABEL_TRANSLATION_DIRECTORIES = 'translations'
    
    # Privacy & GDPR Settings
    CONTACT_PRIVACY_ENABLED = True  # Hide contact details until match confirmed
    CONTACT_BLUR_LEVEL = 'full'  # Options: 'full', 'partial', 'none'
    DATA_RETENTION_DAYS = 365  # Auto-archive items after 1 year
    ALLOW_DATA_EXPORT = True  # Users can export their data (GDPR compliance)