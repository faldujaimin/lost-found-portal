from flask import Flask, render_template, redirect, url_for, flash, request, jsonify, send_file, make_response
from flask import abort
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import check_password_hash
from passlib.context import CryptContext
from datetime import datetime, timedelta, date
import os
import sqlite3
import shutil
from datetime import datetime as _dt
from sqlalchemy import event, func, extract
from sqlalchemy.engine import Engine
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TextAreaField, DateField, FileField, BooleanField
from wtforms.validators import DataRequired, Length, EqualTo, ValidationError, Regexp, Optional
from wtforms.widgets import DateInput
from flask_wtf.file import FileAllowed
from config import Config
import json
import io
import csv
from collections import defaultdict, Counter
try:
    from flask_babel import Babel, gettext as _, lazy_gettext as _l, format_date, format_datetime
    BABEL_AVAILABLE = True
except ImportError:
    BABEL_AVAILABLE = False
    print("WARNING: Flask-Babel not installed. Multi-language support disabled.")
    # Fallback: define dummy functions
    def _(s, **kwargs): return s % kwargs if kwargs else s
    def _l(s): return s
    def format_date(d, format=None): return d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)
    def format_datetime(dt, format=None): return dt.strftime('%Y-%m-%d %H:%M:%S') if hasattr(dt, 'strftime') else str(dt)



try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("WARNING: ReportLab not installed. PDF export disabled.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("WARNING: Pandas not installed. Advanced analytics disabled.")

"""
Lost & Found Portal (Flask)

Overview and key behaviors (for future maintainers):

- Roles: users have a `role` field with one of: 'student', 'admin', 'hod'.
    - Students register via the `/register` form and are always created with role='student'.
    - Admins and HODs have elevated privileges (delete items, view history).

- Item lifecycle: items are reported as 'Lost' or 'Found'. Items are soft-deleted
    by setting `is_active=False` and recording `deleted_by` and `deleted_at`.
    Soft-deletion keeps the record for audit/history rather than removing it.

- Audit logging: actions like reporting and deletion are recorded in the
    `ReportLog` model. That provides an audit trail (user, action, item, timestamp,
    details) and can be viewed at `/debug/logs` (for dev; secure in production).

- History access: viewing the history page requires two levels of checks:
    1) The user must be an 'admin' or 'hod'.
    2) A secondary password (config `HISTORY_PASSWORD`) must be entered.
         This provides an extra gate so not every elevated user can view history
         without the shared secret. The default is a placeholder and should be
         set via environment variable in production.

Security notes:
- Do not use the development secret values in production. Use environment
    variables or a secrets manager for `SECRET_KEY` and `HISTORY_PASSWORD`.
- Debug routes like `/debug/users` and `/debug/logs` are intended for
    development only — lock or remove them in production.
"""

# --- Configuration ---
# Define UPLOAD_FOLDER as an ABSOLUTE path to avoid Current Working Directory issues.
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# Use an explicit, persistent database file inside the instance/ folder so the
# SQLite file remains on disk across server restarts and is easy to back up.
INSTANCE_DIR = os.path.join(PROJECT_ROOT, 'instance')
if not os.path.exists(INSTANCE_DIR):
    try:
        os.makedirs(INSTANCE_DIR)
        print(f"Startup DEBUG: Created instance directory: {INSTANCE_DIR}")
    except OSError as e:
        print(f"Startup ERROR: Failed to create instance directory {INSTANCE_DIR}: {e}")
        # Proceeding may still work if the folder already exists or is created later.

DB_FILENAME = 'site.db'
DB_PATH = os.path.join(INSTANCE_DIR, DB_FILENAME)
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Items that should require explicit confirmation when their name appears in the report
PROTECTED_KEYWORDS = [
    'phone','mobile','cell phone','iphone','android','samsung','xiaomi','pixel','phone','handphone',
    'book','bag','backpack','purse','wallet','keys','key','id','id card','card',
    'laptop','macbook','notebook','watch','ring','jewel','jewelry','glasses','specs','spectacles',
    'umbrella','pen','pens','pen drive','pendrive','usb','flash drive','calculator','calc','charger',
    'cable','wire','bottle','water bottle','mouse','file','files','document','documents','chain',
    'necklace','handsfree','hands-free','headphone','headphones','earphones','earbuds','head set','head-set'
]
# Items that should trigger strict rejection if the photo is bad/distorted
PROTECTED_STRICT = [
    'phone', 'mobile', 'cell phone', 'iphone', 'android', 'samsung', 'xiaomi', 'pixel',
    'laptop', 'macbook', 'notebook', 'wallet', 'purse', 'id card', 'card', 'id'
]

app = Flask(__name__)
app.config.from_object(Config)

# Check environmental toggle for AI features (default to True)
# PRO TIP: Set this to 'false' on Render free tier to save memory!
ENABLE_AI = os.environ.get('ENABLE_AI_FEATURES', 'true').lower() == 'true'
if not ENABLE_AI:
    print("WARNING: AI features are DISABLED by configuration (ENABLE_AI_FEATURES=false)")
else:
    print("INFO: AI features are ENABLED. Models will load on demand.")

# Register Jinja2 filters and globals
app.jinja_env.filters['fromjson'] = json.loads
app.jinja_env.globals.update(format_date=format_date, format_datetime=format_datetime)

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
# Note: history page is no longer protected by a secondary password. Access is
# controlled solely by user role (admin or hod).

# Initialize SQLAlchemy with the Flask app instance
db = SQLAlchemy(app)

# Initialize Flask-Babel for multi-language support
if BABEL_AVAILABLE:
    def get_locale():
        """Select language based on user preference or browser settings"""
        # 1. Check if user is logged in and has a preference
        if current_user and hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
            if hasattr(current_user, 'preferred_language') and current_user.preferred_language:
                return current_user.preferred_language
        
        # 2. Check session override
        if 'language' in request.args:
            lang = request.args.get('language')
            if lang in app.config.get('BABEL_SUPPORTED_LOCALES', ['en']):
                from flask import session
                session['language'] = lang
                return lang
        
        from flask import session
        if 'language' in session:
            return session.get('language')
        
        # 3. Auto-detect from browser Accept-Language header
        return request.accept_languages.best_match(app.config.get('BABEL_SUPPORTED_LOCALES', ['en'])) or 'en'

    babel = Babel(app, locale_selector=get_locale)
    
    print("INFO: Flask-Babel initialized for multi-language support")

# Ensure SQLite uses sensible pragmas for durability and concurrency.
# This sets WAL journal mode, enables foreign keys, and sets synchronous to NORMAL
# so commits are durable but not excessively slow. It applies only to SQLite connections.
@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    # Only apply for the sqlite3 DB-API connection object
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            # journal_mode may return the current mode; ignore failures
            pass
        try:
            cursor.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        try:
            cursor.execute("PRAGMA foreign_keys=ON;")
        except Exception:
            pass
        cursor.close()
# Initialize Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = 'login' # Set the login view for @login_required decorator

# Password hashing context: prefer Argon2 and accept pbkdf2_sha256 for legacy hashes
pwd_context = CryptContext(schemes=["argon2", "pbkdf2_sha256"], deprecated="auto")

try:
    from zoneinfo import ZoneInfo
    TZ_INDIA = ZoneInfo('Asia/Kolkata')
except Exception:
    TZ_INDIA = None

def create_notification(user_id, message, link=None, type='info'):
    """Helper to create a systematic notification for a user."""
    try:
        notif = Notification(
            user_id=user_id,
            message=message,
            link=link,
            type=type
        )
        db.session.add(notif)
        db.session.commit()
        return True
    except Exception as e:
        print(f"ERROR: Failed to create notification: {e}")
        db.session.rollback()
        return False

# --- Helper Functions ---
# Checks if an uploaded filename has an allowed extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Optional: server-side verification for protected/valuable items.
# This will try to use torchvision's object detection (COCO) if available.
# If torchvision is not installed, a lightweight heuristic fallback is used
# (checks image dimensions and aspect ratio). The goal is to ensure the
# uploaded image plausibly contains the claimed protected item; otherwise
# the upload is rejected and the user is asked to resend a clearer photo.

# COCO category names (standard list used by torchvision detection models)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Lazy-loaded detection model (if torchvision is available)
_detection_model = None
_detection_device = 'cpu'

def _load_detection_model():
    global _detection_model
    if not ENABLE_AI:
        return None
    if _detection_model is not None:
        return _detection_model
    try:
        import torch
        from torchvision import transforms
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
    except Exception as e:
        print(f"DEBUG: torchvision or torch not available for image verification: {e}")
        return None

    try:
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        model.to(_detection_device)
        model.eval()
        _detection_model = (model, transforms)
        print("DEBUG: Loaded torchvision detection model for image verification.")
        return _detection_model
    except Exception as e:
        print(f"DEBUG: Failed to initialize detection model: {e}")
        return None

# Lazy CLIP loader and verifier (zero-shot image-text matching)
_clip_model = None
_clip_processor = None
_clip_device = 'cpu'

def _load_clip_model():
    global _clip_model, _clip_processor
    if not ENABLE_AI:
        return None
    if _clip_model is not None and _clip_processor is not None:
        return (_clip_model, _clip_processor)
    try:
        from transformers import CLIPProcessor, CLIPModel  # type: ignore[import]
        import torch  # type: ignore[import]
    except Exception as e:
        print(f"DEBUG: CLIP (transformers) not available: {e}")
        return None

    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.to(_clip_device)
        model.eval()
        _clip_model = model
        _clip_processor = processor
        print("DEBUG: Loaded CLIP model for image-text matching.")
        return (_clip_model, _clip_processor)
    except Exception as e:
        print(f"DEBUG: Failed to initialize CLIP model: {e}")
        return None


def clip_verify(item_name, file_path, threshold=None):
    """Return (result, info, best_label, score) where result is True/False/None.
    If CLIP is not available, returns (None, 'clip_unavailable', None, None).
    """
    if threshold is None:
        threshold = app.config.get('CLIP_THRESHOLD', 0.22)
    model_proc = _load_clip_model()
    if not model_proc:
        return (None, 'clip_unavailable', None, None)
    model, processor = model_proc
    try:
        from PIL import Image
        import torch
        img = Image.open(file_path).convert('RGB')
        # Build candidates: the item's name and some short variants
        name = (item_name or '').strip()
        candidates = [name]
        # split words to produce shorter tokens like 'phone', 'wallet' etc.
        for tok in name.replace('-', ' ').replace('/', ' ').split():
            if tok and tok.lower() not in candidates:
                candidates.append(tok)
        
        # Add a set of common object classes to serve as negatives/comparators
        # Refined to avoid competition with claimed item's category
        common_objects = [
            'cell phone', 'laptop', 'backpack', 'bag', 'wallet', 
            'keys', 'watch', 'headphones', 'bottle', 'umbrella',
            'id card', 'glasses', 'book', 'shoe', 'clothing', 
            'charger', 'calculator', 'trash', 'floor', 'hand'
        ]
        
        # Determine claimed categories to filter synonyms from negatives
        object_categories = {
            'phone': ['phone', 'mobile', 'cell', 'smartphone', 'iphone', 'android'],
            'laptop': ['laptop', 'computer', 'notebook', 'macbook'],
            'bag': ['bag', 'backpack', 'handbag', 'purse', 'suitcase', 'luggage', 'tote'],
            'wallet': ['wallet', 'purse'],
            'watch': ['watch', 'clock', 'wristwatch', 'smartwatch', 'iwatch'],
            'keys': ['keys', 'key'],
            'card': ['card', 'id', 'license', 'student id'],
            'book': ['book', 'notebook', 'textbook', 'folder'],
            'glasses': ['glasses', 'sunglasses', 'spectacles']
        }
        
        claimed_cats = set()
        for cat, kws in object_categories.items():
            if any(kw in name.lower() for kw in kws):
                claimed_cats.add(cat)
        
        for obj in common_objects:
            # Skip negatives that belong to the same category as the claimed item
            is_synonym = False
            for cat in claimed_cats:
                if any(kw in obj for kw in object_categories[cat]):
                    is_synonym = True
                    break
            
            if not is_synonym and obj not in candidates and obj not in name.lower():
                candidates.append(obj)

        inputs = processor(text=candidates, images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
        
        # Get score for the user's item name (or best matching part of it)
        user_candidate_indices = [i for i, c in enumerate(candidates) if c.lower() in name.lower() or name.lower() in c.lower()]
        
        if not user_candidate_indices:
             user_candidate_indices = [0]
             
        # Sum probabilities of all candidates that match the item name
        # This handles 'bag' and 'gray bag' both being candidates
        claimed_score = sum([float(probs[i]) for i in user_candidate_indices])
        
        # find best match overall
        best_idx = int(probs.argmax())
        best_score = float(probs[best_idx])
        best_label = candidates[best_idx]
        
        print(f"DEBUG: CLIP: best match '{best_label}' score={best_score:.3f}, claimed '{item_name}' score={claimed_score:.3f}")
        
        # We return the score of the CLAIMED item
        # Relaxed threshold: if claimed score is decent OR it's the best match
        clip_threshold = threshold
        if claimed_score >= clip_threshold or (best_idx in user_candidate_indices):
            return (True, f"CLIP matched '{name}' ({claimed_score:.2f})", best_label, claimed_score)
        else:
            return (False, f"CLIP found '{best_label}' ({best_score:.2f}) instead of '{name}'", best_label, claimed_score)
            
    except Exception as e:
        print(f"DEBUG: CLIP verification failed: {e}")
        return (None, f"clip_error: {e}", None, None)

@app.route('/api/analyze_image', methods=['POST'])
def api_analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        # Save temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_' + filename)
        file.save(temp_path)
        
        results = {
            'detected_objects': [],
            'best_guess': None,
            'confidence': 0.0
        }
        
        # 1. Resize image for faster analysis (Speed Optimization)
        try:
            from PIL import Image
            img_for_resize = Image.open(temp_path).convert('RGB')
            if max(img_for_resize.size) > 800:
                img_for_resize.thumbnail((800, 800), Image.Resampling.LANCZOS)
                img_for_resize.save(temp_path, quality=85)
            img_for_resize.close()
        except Exception as e:
            print(f"Resize optimization failed: {e}")

        # 2. Run YOLO (Lower threshold for better recall/accuracy)
        try:
            yolo_dets = detect_with_yolo(temp_path, conf=0.3)
            # Filter out 'person' as it's often background noise in these photos
            yolo_dets = [d for d in yolo_dets if d['label'] != 'person']
            
            results['detected_objects'] = yolo_dets
            if yolo_dets:
                # If YOLO is very confident, use it immediately
                # EXCEPT for certain categories known for false positives (like scissors)
                if yolo_dets[0]['score'] > 0.8 and yolo_dets[0]['label'] not in ['scissors', 'handbag']:
                    results['best_guess'] = yolo_dets[0]['label']
                    results['confidence'] = yolo_dets[0]['score']
                    results['source'] = 'yolo'
        except Exception as e:
            print(f"YOLO failed: {e}")
            
        # 3. Use CLIP if YOLO is uncertain or empty (Accuracy Optimization)
        try:
             if not results['best_guess'] or results['confidence'] < 0.7:
                 # Simplified CLIP classification
                 model_proc = _load_clip_model()
                 if model_proc:
                     model, processor = model_proc
                     from PIL import Image
                     import torch
                     img = Image.open(temp_path).convert('RGB')
                     
                     common_objects = [
                        'cell phone', 'smartphone', 'iphone', 'android phone',
                        'laptop', 'macbook', 'computer',
                        'bag', 'backpack', 'handbag', 'tote bag',
                        'wallet', 'purse', 'pouch',
                        'watch', 'smartwatch', 'smart watch', 'digital watch', 'wrist watch',
                        'keys', 'keychain', 'car keys',
                        'headphones', 'earbuds', 'airpods',
                        'water bottle', 'bottle', 'flask', 'thermos',
                        'id card', 'student id', 'credit card', 'license',
                        'book', 'textbook', 'notebook', 'folder', 'file',
                        'charger', 'adapter', 'usb cable', 'cable', 'wire',
                        'glasses', 'sunglasses', 'spectacles',
                        'umbrella', 'mouse', 'keyboard', 'calculator',
                        'pen', 'pencil', 'stationery',
                        'hat', 'cap', 'clothing', 'shoe', 'sneaker',
                        'ring', 'bracelet', 'necklace', 'jewelry',
                        'powerbank', 'pendrive', 'flash drive'
                     ]
                     inputs = processor(text=common_objects, images=img, return_tensors="pt", padding=True)
                     with torch.no_grad():
                        outputs = model(**inputs)
                        probs = outputs.logits_per_image.softmax(dim=1)[0]
                        
                     best_idx = int(probs.argmax())
                     best_label = common_objects[best_idx]
                     score = float(probs[best_idx])
                     
                     if score > 0.3:
                         results['best_guess'] = best_label
                         results['confidence'] = score
                         results['source'] = 'clip'
        except Exception as e:
            print(f"CLIP analysis failed: {e}")
            
        return jsonify(results)
    except Exception as e:
        print(f"ERROR: api_analyze_image failed: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleanup
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass



# Lazy YOLOv8 loader and detector (Ultralytics)
_yolo_model = None

def _load_yolo():
    global _yolo_model
    if not ENABLE_AI:
        return None
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"DEBUG: ultralytics YOLO not available: {e}")
        return None
    try:
        model = YOLO('yolov8n.pt')
        _yolo_model = model
        print("DEBUG: Loaded YOLOv8 model (yolov8n).")
        return _yolo_model
    except Exception as e:
        print(f"DEBUG: Failed to initialize YOLO model: {e}")
        return None


def detect_with_yolo(image_path, conf=0.5):
    """Run YOLOv8 detection and return list of {'label','score','box','area'} sorted by score desc."""
    model = _load_yolo()
    if not model:
        return []
    try:
        results = model(image_path, conf=conf)
        out = []
        for r in results:
            boxes = getattr(r, 'boxes', [])
            for box in boxes:
                try:
                    cls = int(box.cls[0])
                    score = float(box.conf[0])
                    name = model.names.get(cls, str(cls)).lower()
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    area = (x2 - x1) * (y2 - y1)
                    out.append({'label': name, 'score': score, 'box': (x1, y1, x2, y2), 'area': area})
                except Exception:
                    continue
        out.sort(key=lambda d: d['score'], reverse=True)
        print(f"DEBUG: YOLO detections: {out}")
        return out
    except Exception as e:
        print(f"DEBUG: YOLO detection failed: {e}")
        return []


# Lazy EasyOCR loader for text extraction
_ocr_reader = None

def _load_ocr_reader():
    """Lazy load EasyOCR reader (downloads model on first use)."""
    global _ocr_reader
    if not ENABLE_AI:
        return None
    if _ocr_reader is not None:
        return _ocr_reader
    try:
        import easyocr
    except Exception as e:
        print(f"DEBUG: EasyOCR not available: {e}")
        return None
    try:
        # Initialize with English language support
        # Add more languages as needed: ['en', 'hi', 'mr', 'ta', etc.]
        reader = easyocr.Reader(['en'], gpu=False)  # Use CPU for compatibility
        _ocr_reader = reader
        print("DEBUG: Loaded EasyOCR reader for text extraction.")
        return _ocr_reader
    except Exception as e:
        print(f"DEBUG: Failed to initialize EasyOCR reader: {e}")
        return None


def extract_text_from_image(image_path, languages=['en']):
    """
    Extract text from image using EasyOCR.
    Returns: (success: bool, extracted_text: str, raw_results: list)
    """
    reader = _load_ocr_reader()
    if not reader:
        return (False, "", [])
    
    try:
        # Read text from image
        results = reader.readtext(image_path)
        
        # Extract text strings and confidence scores
        extracted_lines = []
        all_text = []
        
        for (bbox, text, confidence) in results:
            # Only include text with confidence > 0.3
            if confidence > 0.3:
                extracted_lines.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })
                all_text.append(text)
        
        # Combine all extracted text
        combined_text = ' '.join(all_text)
        
        print(f"DEBUG: OCR extracted {len(extracted_lines)} text segments from image")
        print(f"DEBUG: Combined text: {combined_text[:200]}...")  # Print first 200 chars
        
        return (True, combined_text, extracted_lines)
    except Exception as e:
        print(f"DEBUG: OCR text extraction failed: {e}")
        return (False, "", [])


def suggest_item_name(extracted_text, max_suggestions=3):
    """
    Analyze extracted text and suggest possible item names.
    Returns: list of suggested item names
    """
    if not extracted_text or not extracted_text.strip():
        return []
    
    suggestions = []
    text_lower = extracted_text.lower()
    
    # Common patterns for different item types
    patterns = {
        # Phone brands and models
        'phone': ['iphone', 'samsung', 'xiaomi', 'redmi', 'oppo', 'vivo', 'oneplus', 'pixel', 'galaxy', 'mi ', 'realme', 'motorola', 'nokia'],
        # Laptop/computer brands
        'laptop': ['macbook', 'dell', 'hp', 'lenovo', 'asus', 'acer', 'thinkpad', 'latitude', 'inspiron'],
        # ID cards
        'id_card': ['student id', 'identity', 'id card', 'identification', 'college id', 'university'],
        # Books
        'book': ['isbn', 'edition', 'author', 'publisher', 'copyright'],
        # Other items
        'wallet': ['wallet', 'cardholder', 'purse'],
        'watch': ['watch', 'time', 'clock', 'smartwatch', 'fitbit', 'garmin'],
        'bottle': ['bottle', 'flask', 'tumbler', 'hydro'],
        'charger': ['charger', 'adapter', 'power', 'usb-c', 'lightning'],
    }
    
    # Check for ID card patterns
    if any(keyword in text_lower for keyword in patterns['id_card']):
        # Try to extract name from ID card
        words = extracted_text.split()
        # Look for capitalized words that might be names
        name_candidates = [w for w in words if w and w[0].isupper() and len(w) > 2]
        if name_candidates:
            suggestions.append(f"Student ID Card - {' '.join(name_candidates[:2])}")
        else:
            suggestions.append("Student ID Card")
    
    # Check for phone patterns
    phone_match = None
    for keyword in patterns['phone']:
        if keyword in text_lower:
            # Try to extract model info
            idx = text_lower.find(keyword)
            surrounding = extracted_text[max(0, idx-10):min(len(extracted_text), idx+50)]
            phone_match = surrounding.strip()
            break
    if phone_match:
        suggestions.append(f"Mobile Phone - {phone_match}")
    
    # Check for laptop patterns
    laptop_match = None
    for keyword in patterns['laptop']:
        if keyword in text_lower:
            idx = text_lower.find(keyword)
            surrounding = extracted_text[max(0, idx-10):min(len(extracted_text), idx+40)]
            laptop_match = surrounding.strip()
            break
    if laptop_match:
        suggestions.append(f"Laptop - {laptop_match}")
    
    # Check for book patterns
    if any(keyword in text_lower for keyword in patterns['book']):
        # Try to find title (usually the longest capitalized phrase)
        words = extracted_text.split()
        title_words = []
        for w in words:
            if w and (w[0].isupper() or w.isupper()) and len(w) > 2:
                title_words.append(w)
            elif title_words:
                break  # Stop at first non-capitalized word after finding title
        if title_words:
            suggestions.append(f"Book - {' '.join(title_words[:5])}")
        else:
            suggestions.append("Book")
    
    # Check for other items
    for item_type, keywords in patterns.items():
        if item_type not in ['phone', 'laptop', 'id_card', 'book']:
            if any(keyword in text_lower for keyword in keywords):
                suggestions.append(item_type.replace('_', ' ').title())
    
    # If no specific patterns matched, extract most prominent text
    if not suggestions and extracted_text:
        # Get the longest sequence of capitalized words
        words = extracted_text.split()
        longest_seq = []
        current_seq = []
        for w in words:
            if w and len(w) > 2 and (w[0].isupper() or w.isupper()):
                current_seq.append(w)
            else:
                if len(current_seq) > len(longest_seq):
                    longest_seq = current_seq
                current_seq = []
        if len(current_seq) > len(longest_seq):
            longest_seq = current_seq
        
        if longest_seq:
            suggestions.append(' '.join(longest_seq[:4]))
    
    # Return top suggestions
    return suggestions[:max_suggestions]


# Mapping from common keywords to COCO labels we expect to see in an image
# For many college items COCO doesn't have an exact class; those will use stricter heuristics.
KEYWORD_TO_COCO_LABELS = {
    'phone': ['cell phone'],
    'mobile': ['cell phone'],
    'iphone': ['cell phone'],
    'android': ['cell phone'],
    'samsung': ['cell phone'],
    'xiaomi': ['cell phone'],
    'pixel': ['cell phone'],
    'handphone': ['cell phone'],
    'laptop': ['laptop'],
    'macbook': ['laptop'],
    'notebook': ['laptop'],
    'backpack': ['backpack'],
    'bag': ['handbag', 'backpack'],
    'purse': ['handbag'],
    'wallet': ['handbag'],
    'bottle': ['bottle'],
    'mouse': ['mouse'],
    'umbrella': ['umbrella'],
    'book': ['book'],
    'keys': [],
    'key': [],
    'id': [],
    'id card': [],
    'card': [],
    'watch': ['watch', 'clock'],
    'smartwatch': ['watch', 'clock'],
    'smart watch': ['watch', 'clock'],
    'iwatch': ['watch', 'clock'],
    'apple watch': ['watch', 'clock'],
    'ring': [],
    'glasses': ['glasses', 'sunglasses'],
    'specs': ['glasses', 'sunglasses'],
    'spectacles': [],
    'charger': [],
    'cable': [],
    'wire': [],
    'pen': [],
    'pendrive': [],
    'pen drive': [],
    'flash drive': [],
    'earphones': [],
    'earbuds': [],
    'headphone': [],
    'headphones': [],
    'file': [],
    'files': [],
    'document': [],
    'documents': [],
    'necklace': [],
    'jewelry': [],
    'calculator': [],
}

# Treat all configured protected keywords as items that need stricter verification
PROTECTED_STRICT = set([k.lower() for k in PROTECTED_KEYWORDS])
# Hardcoded set was dead code, now merged into the logic if needed or just kept as is.
# PROTECTED_STRICT.update(['laptop','macbook','notebook','wallet','id','id card','card','backpack','bag','purse'])



def verify_protected_image(item_name, file_path, score_threshold=0.3, min_size=120, yolo_dets=None):
    """Return (True, info) if image plausibly contains the claimed item.
    If False, info contains a short reason or detected labels.

    Behavior:
    - Lowered confidence threshold to 0.3 for small/cropped photos.
    - If COCO detection is available and expected labels exist, try detection first.
    - If detection fails to find expected labels, do NOT immediately reject — fall back to heuristic checks.
    - Only reject if both detection (if used) and heuristic checks fail.
    - Print detailed DEBUG info to help diagnose false negatives.
    """
    name = (item_name or '').lower()
    # Find which keywords from our mapping appear in the item name
    matched_keywords = [k for k in KEYWORD_TO_COCO_LABELS.keys() if k in name]
    if not matched_keywords:
        return (True, 'No protected keywords matched; no verification needed')

    # Build expected labels from mapping (may be empty)
    expected = set()
    for k in matched_keywords:
        for lab in KEYWORD_TO_COCO_LABELS.get(k, []):
            if lab:
                expected.add(lab.lower())

    detection_attempted = False
    detection_matched = False
    detection_info = None

    # First try YOLOv8 (preferred if available) - this helps detect many object classes reliably
    try:
        if yolo_dets is None:
            yolo_dets = detect_with_yolo(file_path)
        
        if yolo_dets:
            detection_attempted = True
            print(f"DEBUG: YOLO labels for '{item_name}': {yolo_dets}")
            # If YOLO finds a matching expected label, accept
            for exp in expected:
                for d in yolo_dets:
                    if exp in d['label'] or d['label'] in exp:
                        return (True, f"YOLO detected: {d['label']} ({d['score']:.2f})")
            # If YOLO top detection is a strong conflicting object, perform a CLIP "sanity check" before rejecting
            top = yolo_dets[0]
            conflict_thresh = app.config.get('DETECTION_CONFLICT_THRESHOLD', 0.7)
            if top['score'] >= conflict_thresh:
                # Sanity Check: If CLIP thinks it's the item_name, don't reject based on YOLO conflict
                try:
                    clip_ok, _, _, _ = clip_verify(item_name, file_path)
                    if clip_ok:
                        print(f"DEBUG: YOLO conflict ('{top['label']}') overridden by CLIP for '{item_name}'")
                        return (True, f"YOLO detected {top['label']} but CLIP verified as {item_name}")
                except Exception:
                    pass

                reason = f"Uploaded image appears to show '{top['label']}' (confidence {top['score']:.2f}), not '{item_name}'. Please upload a photo of the claimed item or correct the item name."
                print(f"DEBUG: YOLO detection conflict (confirmed by lack of CLIP match): {reason}")
                return (False, reason)
            # Otherwise continue to other checks (torchvision/CLIP/heuristic)
    except Exception as e:
        print(f"DEBUG: YOLO verification step failed: {e}")

    # Try to run object detection if possible **and** we have expected labels
    model_tuple = _load_detection_model()
    if model_tuple and expected:
        detection_attempted = True
        try:
            import torch
            from PIL import Image
            model, transforms = model_tuple
            img = Image.open(file_path).convert('RGB')
            transform = transforms.Compose([transforms.ToTensor()])
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                outputs = model(tensor)[0]

            detected = []
            for lbl_idx, score in zip(outputs['labels'].tolist(), outputs['scores'].tolist()):
                if score < score_threshold:
                    continue
                # Defensive: ensure index is in range
                if lbl_idx < 0 or lbl_idx >= len(COCO_INSTANCE_CATEGORY_NAMES):
                    continue
                lbl = COCO_INSTANCE_CATEGORY_NAMES[lbl_idx].lower()
                detected.append((lbl, float(score)))

            # sort detections by score desc
            detected.sort(key=lambda x: x[1], reverse=True)
            detected_labels = [d[0] for d in detected]
            detection_info = detected
            print(f"DEBUG: Detection results for '{item_name}': {detected}")

            # Substring match: accept if any expected label appears within any detected label
            for exp in expected:
                for dl in detected_labels:
                    if exp in dl or dl in exp:
                        detection_matched = True
                        break
                if detection_matched:
                    break

            if detection_matched:
                return (True, f"Detected labels: {detected}")
            else:
                # If detection found a strong, conflicting object, reject immediately
                if detected:
                    top_label, top_score = detected[0]
                    conflict_thresh = app.config.get('DETECTION_CONFLICT_THRESHOLD', 0.7)
                    if top_score >= conflict_thresh:
                        reason = f"Uploaded image appears to show '{top_label}' (confidence {top_score:.2f}), not '{item_name}'. Please upload a photo of the claimed item or correct the item name."
                        print(f"DEBUG: Detection conflict: {reason}")
                        return (False, reason)

                # Do not reject yet; attempt CLIP verification (if enabled) or heuristic fallback below
                print(f"DEBUG: Detection did not find expected labels. Expected: {sorted(expected)} Detected: {detected}")
                # Try CLIP as a semantic fallback if configured
                try:
                    if app.config.get('USE_CLIP_VERIFICATION', True):
                        c_ok, c_info, c_best, c_score = clip_verify(item_name, file_path)
                        print(f"DEBUG: CLIP result for '{item_name}': {c_ok}, info={c_info}, best={c_best}, score={c_score}")
                        if c_ok is True:
                            return (True, f"CLIP match: {c_info}")
                        elif c_ok is False:
                            # If CLIP strongly indicates a different object, reject
                            clip_conflict_thresh = app.config.get('CLIP_CONFLICT_THRESHOLD', 0.6)
                            normalized_c_best = (c_best or '').lower()
                            # build claim tokens from matched keywords and item name words
                            claim_tokens = set(k.lower() for k in matched_keywords)
                            for tok in name.replace('-', ' ').replace('/', ' ').split():
                                if tok:
                                    claim_tokens.add(tok.lower())

                            expected_lower = set(e.lower() for e in expected)

                            # check if CLIP best label matches any expected or claim token
                            match_found = False
                            for exp in expected_lower:
                                if exp and (exp in normalized_c_best or normalized_c_best in exp):
                                    match_found = True
                                    break
                            if not match_found:
                                for tok in claim_tokens:
                                    if tok and (tok in normalized_c_best or normalized_c_best in tok):
                                        match_found = True
                                        break

                            if c_score is not None and c_score >= clip_conflict_thresh and not match_found:
                                reason = f"Uploaded image seems to show '{c_best}' (CLIP score {c_score:.2f}), not '{item_name}'. Please upload a photo of the claimed item or correct the item name."
                                print(f"DEBUG: CLIP conflict: {reason}")
                                return (False, reason)

                            # CLIP didn't match strongly (or matched a synonym); record info and continue to heuristic
                            detection_info = f"clip_no_match: {c_info}"
                        else:
                            # CLIP unavailable or errored; note it
                            detection_info = f"clip_unavailable_or_error: {c_info}"
                except Exception as e:
                    print(f"DEBUG: CLIP verification raised an exception: {e}")
                    detection_info = f"clip_error: {e}"
        except Exception as e:
            print(f"DEBUG: Detection failed during verify: {e}")
            detection_info = f"error: {e}"
            # Fall through to heuristic check

    # Heuristic fallback: check image dimensions and basic aspect ratio
    try:
        from PIL import Image, ImageFilter
        img = Image.open(file_path)
        w, h = img.size

        # Strict items get a tougher heuristic if detection was unavailable/failed
        is_strict = any(k in name for k in PROTECTED_STRICT)
        if is_strict:
            # Per-item minimums for the *longer* side (allows portrait phone photos like 155x324)
            ITEM_MIN_LONG_SIDE = {
                'phone': 300,
                'mobile': 300,
                'iphone': 300,
                'android': 300,
                'samsung': 300,
                'xiaomi': 300,
                'pixel': 300,
                'handphone': 300,
                'laptop': 300,
                'macbook': 300,
                'notebook': 300,
                'wallet': 220,
                'backpack': 220,
                'bag': 220,
                'purse': 220,
                'book': 220,
                'bottle': 200,
                'mouse': 200,
                'umbrella': 220,
                'keys': 150,
                'key': 150,
                'id': 200,
                'id card': 200,
                'card': 200,
                'watch': 150,
                'ring': 120,
                'charger': 150,
                'cable': 150,
                'pen': 120,
                'pendrive': 150,
                'pen drive': 150,
                'flash drive': 150,
                'earphones': 150,
                'earbuds': 150,
                'headphone': 150,
                'headphones': 150,
                'file': 200,
                'files': 200,
                'document': 200,
                'documents': 200,
                'necklace': 120,
                'jewelry': 120,
            }
            # Determine the most strict requirement among keywords present
            min_strict_candidates = [ITEM_MIN_LONG_SIDE.get(k, 200) for k in matched_keywords]
            min_strict = max(min_strict_candidates) if min_strict_candidates else 200

            long_side = max(w, h)
            short_side = min(w, h)
            # Require the long side to be at least min_strict
            if long_side < min_strict:
                reason = f"Image too small for verification ({w}x{h}); please upload a clearer close-up where the longer side is at least {min_strict}px (or choose a higher-resolution photo)."
                print(f"DEBUG: Strict heuristic rejection: {reason} (matched_keywords={matched_keywords}, detection_attempted={detection_attempted}, detection_info={detection_info})")
                return (False, reason)
            aspect = max(w/h, h/w)
            if aspect > 3:
                reason = f"Unusual aspect ratio ({w}x{h}); please upload a portrait/close-up photo of the claimed item."
                print(f"DEBUG: Strict heuristic rejection: {reason} (matched_keywords={matched_keywords}, detection_attempted={detection_attempted}, detection_info={detection_info})")
                return (False, reason)

            # Edge/clarity check: ensure the photo is not blank or too blurry
            try:
                # Resize preserving orientation: make longer side min(300, min_strict) for edge check
                resize_long = min(300, min_strict)
                if w >= h:
                    small = img.copy().resize((resize_long, int(resize_long * (h/w)))) if w and h else img.copy()
                else:
                    small = img.copy().resize((int(resize_long * (w/h)), resize_long)) if w and h else img.copy()
                edges = small.convert('L').filter(ImageFilter.FIND_EDGES)
                arr = edges.point(lambda p: 1 if p > 30 else 0)
                edge_count = sum(arr.getdata())
                edge_density = edge_count / (arr.size[0] * arr.size[1])
                print(f"DEBUG: Strict edge density for '{item_name}': {edge_density:.4f}")
                if edge_density < 0.015:
                    reason = f"Image appears blurry or lacks detail; please upload a clearer close-up photo."
                    print(f"DEBUG: Strict heuristic rejection (low edge density): {reason} (matched_keywords={matched_keywords}, detection_attempted={detection_attempted}, detection_info={detection_info})")
                    return (False, reason)
            except Exception as e:
                print(f"DEBUG: Edge check failed in strict heuristic: {e}")
                # fallthrough to accept if edges can't be computed

            print(f"DEBUG: Strict heuristic accepted image ({w}x{h}). (matched_keywords={matched_keywords}, detection_attempted={detection_attempted}, detection_info={detection_info})")
            return (True, f"Strict heuristic OK ({w}x{h})")

        # Non-strict items: looser checks
        if w < min_size or h < min_size:
            reason = f"Image too small ({w}x{h}); please upload a clearer photo at least {min_size}px on each side."
            print(f"DEBUG: Heuristic rejection: {reason} (detection_attempted={detection_attempted}, detection_info={detection_info})")
            return (False, reason)
        aspect = max(w/h, h/w)
        if aspect > 6:
            reason = f"Unusual aspect ratio ({w}x{h}); please upload a portrait/close-up photo of the item."
            print(f"DEBUG: Heuristic rejection: {reason} (detection_attempted={detection_attempted}, detection_info={detection_info})")
            return (False, reason)
        # Heuristic OK — accept even if detection didn't match
        print(f"DEBUG: Heuristic accepted image ({w}x{h}). (detection_attempted={detection_attempted}, detection_info={detection_info})")
        return (True, f"Heuristic OK ({w}x{h})")
    except Exception as e:
        print(f"DEBUG: Heuristic verification failed: {e}")
        # In case of unexpected errors, be permissive rather than blocking users
        return (True, 'Unable to verify image server-side; please ensure the uploaded image clearly shows the item.')


def cross_verify_item(item_name, file_path, extracted_text=None, yolo_dets=None):
    """
    Cross-verify item name against multiple analysis methods:
    1. OCR extracted text (fuzzy matching)
    2. YOLO object detection
    3. CLIP semantic similarity
    
    Returns: (verified: bool, confidence_score: float, details: dict)
    """
    import json
    from difflib import SequenceMatcher
    
    details = {
        'ocr_match': None,
        'yolo_match': None,
        'clip_match': None,
        'ocr_score': 0.0,
        'yolo_score': 0.0,
        'clip_score': 0.0,
        'overall_confidence': 0.0,
        'verification_method': 'multi-modal'
    }
    
    item_name_lower = (item_name or '').lower().strip()
    if not item_name_lower:
        return (False, 0.0, details)
    
    # 1. OCR Text Matching (if text was extracted)
    ocr_weight = 0.3
    if extracted_text and extracted_text.strip():
        extracted_lower = extracted_text.lower()
        
        # Direct substring match
        if item_name_lower in extracted_lower or extracted_lower in item_name_lower:
            details['ocr_score'] = 1.0
            details['ocr_match'] = 'direct_match'
        else:
            # Fuzzy matching for typos/OCR errors
            # Split into words and find best match
            item_words = item_name_lower.split()
            extracted_words = extracted_lower.split()
            
            max_similarity = 0.0
            for item_word in item_words:
                if len(item_word) < 3:
                    continue  # Skip very short words
                for extracted_word in extracted_words:
                    similarity = SequenceMatcher(None, item_word, extracted_word).ratio()
                    max_similarity = max(max_similarity, similarity)
            
            details['ocr_score'] = max_similarity
            details['ocr_match'] = f'fuzzy_match ({max_similarity:.2f})'
        
        print(f"DEBUG: OCR verification - Score: {details['ocr_score']:.2f}, Match: {details['ocr_match']}")
    else:
        details['ocr_match'] = 'no_text_extracted'
        print("DEBUG: OCR verification skipped - no text extracted")
    
    # 2. YOLO Object Detection Matching
    yolo_weight = 0.35
    try:
        if yolo_dets is None:
            yolo_dets = detect_with_yolo(file_path)
            
        if yolo_dets:
            # Check if any YOLO detection matches item name
            best_yolo_score = 0.0
            best_yolo_label = None
            
            for detection in yolo_dets:
                label = detection['label'].lower()
                score = detection['score']
                
                # Check for keyword matches from item name
                item_keywords = item_name_lower.replace('-', ' ').replace('/', ' ').split()
                for keyword in item_keywords:
                    if len(keyword) < 3:
                        continue
                    if keyword in label or label in keyword:
                        if score > best_yolo_score:
                            best_yolo_score = score
                            best_yolo_label = label
            
            details['yolo_score'] = best_yolo_score
            details['yolo_match'] = best_yolo_label if best_yolo_label else 'no_match'
            details['yolo_detections'] = [{'label': d['label'], 'score': d['score']} for d in yolo_dets[:3]]
            
            print(f"DEBUG: YOLO verification - Score: {details['yolo_score']:.2f}, Best match: {best_yolo_label}")
        else:
            details['yolo_match'] = 'no_detections'
            print("DEBUG: YOLO verification - no detections")
    except Exception as e:
        details['yolo_match'] = f'error: {str(e)[:50]}'
        print(f"DEBUG: YOLO verification error: {e}")
    
    # 3. CLIP Semantic Similarity Matching
    clip_weight = 0.35
    try:
        if app.config.get('USE_CLIP_VERIFICATION', True):
            clip_ok, clip_info, clip_best, clip_similarity = clip_verify(item_name, file_path)
            
            if clip_ok is not None and clip_similarity is not None:
                details['clip_score'] = clip_similarity
                details['clip_match'] = clip_best
                details['clip_info'] = clip_info
                
                print(f"DEBUG: CLIP verification - Score: {details['clip_score']:.2f}, Best match: {clip_best}")
            else:
                details['clip_match'] = 'unavailable'
                print("DEBUG: CLIP verification unavailable")
        else:
            details['clip_match'] = 'disabled'
    except Exception as e:
        details['clip_match'] = f'error: {str(e)[:50]}'
        print(f"DEBUG: CLIP verification error: {e}")
    
    
    # Calculate overall confidence score (weighted average)
    total_weight = 0.0
    weighted_sum = 0.0
    
    # Add penalty for conflicts
    conflict_penalty = 0.0
    
    # Check for YOLO conflicts
    if details.get('yolo_detections'):
        detected_objects = [d['label'].lower() for d in details['yolo_detections']]
        
        # Define object categories for conflict detection
        object_categories = {
            'phone': ['phone', 'cell', 'mobile', 'smartphone', 'iphone', 'android'],
            'laptop': ['laptop', 'computer', 'notebook', 'macbook'],
            'bag': ['bag', 'backpack', 'handbag', 'purse', 'suitcase', 'luggage', 'tote'],
            'wallet': ['wallet', 'purse'],
            'watch': ['watch', 'clock', 'wristwatch', 'smartwatch', 'iwatch'],
            'keys': ['keys', 'key'],
            'card': ['card', 'id', 'license', 'student id'],
            'book': ['book', 'notebook', 'textbook', 'folder'],
            'glasses': ['glasses', 'sunglasses', 'spectacles']
        }
        
        # Determine claimed category
        claimed_category = None
        for category, keywords in object_categories.items():
            if any(kw in item_name_lower for kw in keywords):
                claimed_category = category
                break
        
        # Check if detected objects conflict with claimed category
        if claimed_category:
            detected_category = None
            # Get all categories matched by claimed keywords to handle overlaps (like purse = bag/wallet)
            claimed_keyword_categories = set()
            for cat, kws in object_categories.items():
                if any(kw in item_name_lower for kw in kws):
                    claimed_keyword_categories.add(cat)

            for obj in detected_objects:
                obj_categories = set()
                for cat, kws in object_categories.items():
                    if any(kw in obj for kw in kws):
                        obj_categories.add(cat)
                
                # If the detected object has categories, and NONE of them match ANY of the claimed categories
                # then it's a conflict. (e.g., claimed 'bag', detected 'phone')
                if obj_categories and not (obj_categories & claimed_keyword_categories):
                    # Only penalize if YOLO is confident about this specific conflicting object
                    # and the claimed object was NOT detected with decent confidence
                    yolo_score = details.get('yolo_score', 0)
                    if yolo_score < 0.3: # Claimed object not found or very weak in YOLO
                        conflict_penalty = 0.7
                        details['conflict'] = f"Claimed {claimed_category} but YOLO detected {', '.join(obj_categories)}"
                        print(f"DEBUG: CONFLICT DETECTED - Claimed: {claimed_category}, Detected: {obj_categories}")
                        break
    
    if details['ocr_score'] > 0:
        weighted_sum += details['ocr_score'] * ocr_weight
        total_weight += ocr_weight
    
    if details['yolo_score'] > 0:
        weighted_sum += details['yolo_score'] * yolo_weight
        total_weight += yolo_weight
    
    if details['clip_score'] > 0:
        weighted_sum += details['clip_score'] * clip_weight
        total_weight += clip_weight
    
    # Calculate final confidence
    if total_weight > 0:
        overall_confidence = weighted_sum / total_weight
    else:
        # No detection methods returned scores - use minimum confidence
        overall_confidence = 0.1
    
    # Apply conflict penalty
    overall_confidence = max(0.0, overall_confidence - conflict_penalty)
    
    details['overall_confidence'] = overall_confidence
    details['conflict_penalty'] = conflict_penalty
    
    # Determine if verified (confidence threshold)
    verification_threshold = 0.4  # 40% confidence minimum
    verified = overall_confidence >= verification_threshold
    
    print(f"DEBUG: Cross-verification complete - Confidence: {overall_confidence:.2f}, Verified: {verified}, Conflict Penalty: {conflict_penalty}")
    
    return (verified, overall_confidence, details)


# --- AI Smart Matching Helper ---
def find_smart_matches(target_item):
    """
    Find potential matches for a given item (Lost -> Found or Found -> Lost).
    Returns a list of dicts with {'item', 'score', 'reasons'}.
    """
    if not target_item:
        return []
    
    # We look for opposite status items that are active
    opposite_status = 'Found' if target_item.status == 'Lost' else 'Lost'
    candidates = Item.query.filter_by(status=opposite_status, is_active=True).all()
    
    matches = []
    target_name = target_item.item_name.lower()
    target_tags = set(target_name.split())
    
    for item in candidates:
        score = 0
        reasons = []
        name = item.item_name.lower()
        item_tags = set(name.split())
        
        # 1. Name Similarity
        if target_name == name:
            score += 0.8
            reasons.append(_("Exact name match"))
        else:
            intersection = target_tags.intersection(item_tags)
            if intersection:
                overlap = len(intersection) / max(len(target_tags), len(item_tags))
                if overlap > 0.3:
                    score += 0.5
                    reasons.append(_("Keyword match: %(keywords)s", keywords=', '.join(intersection)))
            
        # 2. Location Match
        if target_item.location and item.location:
            if target_item.location.lower() == item.location.lower():
                score += 0.3
                reasons.append(_("Same location reported"))
            
        # 3. Date Proximity
        if target_item.lost_found_date and item.lost_found_date:
            diff = abs((target_item.lost_found_date - item.lost_found_date).days)
            if diff <= 7:
                date_score = 0.2 * (1 - (diff / 7))
                score += date_score
                reasons.append(_("Dates match within %(diff)d days", diff=diff))

        # 4. AI-Detection Match
        try:
            if target_item.verification_details and item.verification_details:
                target_dets = json.loads(target_item.verification_details).get('yolo_detections', [])
                item_dets = json.loads(item.verification_details).get('yolo_detections', [])
                target_labels = {d['label'] for d in target_dets}
                item_labels = {d['label'] for d in item_dets}
                label_overlap = target_labels.intersection(item_labels)
                if label_overlap:
                    score += 0.4
                    reasons.append(_("AI detected similar objects: %(labels)s", labels=', '.join(label_overlap)))
        except Exception:
            pass
        
        if score >= 0.4:
            matches.append({
                'item': item,
                'score': int(min(score * 100, 100)),
                'reasons': reasons
            })
            
    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches[:5]


# --- Analytics Helper Functions ---
def generate_daily_analytics(target_date=None):
    """Generate analytics data for a specific date (defaults to today)"""
    if target_date is None:
        target_date = date.today()
    
    # Check if analytics already exist for this date
    existing = Analytics.query.filter_by(date=target_date).first()
    if existing:
        return existing
    
    # Calculate date range
    start_datetime = datetime.combine(target_date, datetime.min.time())
    end_datetime = datetime.combine(target_date, datetime.max.time())
    
    # Count items reported on this date
    items_lost = Item.query.filter(
        Item.status == 'Lost',
        Item.reported_at >= start_datetime,
        Item.reported_at <= end_datetime
    ).count()
    
    items_found = Item.query.filter(
        Item.status == 'Found',
        Item.reported_at >= start_datetime,
        Item.reported_at <= end_datetime
    ).count()
    
    items_returned = Item.query.filter(
        Item.status == 'Reclaimed',
        Item.reported_at >= start_datetime,
        Item.reported_at <= end_datetime
    ).count()
    
    # Count new users
    new_users = User.query.filter(
        User.created_at >= start_datetime,
        User.created_at <= end_datetime
    ).count() if hasattr(User, 'created_at') else 0
    
    # Count active users (logged in today)
    active_users = AuthLog.query.filter(
        AuthLog.action == 'login',
        AuthLog.timestamp >= start_datetime,
        AuthLog.timestamp <= end_datetime
    ).distinct(AuthLog.user_id).count()
    
    total_logins = AuthLog.query.filter(
        AuthLog.action == 'login',
        AuthLog.timestamp >= start_datetime,
        AuthLog.timestamp <= end_datetime
    ).count()
    
    # Calculate success rate (all time, up to this date)
    total_items = Item.query.filter(Item.reported_at <= end_datetime).count()
    returned_items = Item.query.filter(
        Item.status == 'Reclaimed',
        Item.reported_at <= end_datetime
    ).count()
    success_rate = (returned_items / total_items * 100) if total_items > 0 else 0.0
    
    # Calculate average return time
    returned_with_dates = Item.query.filter(
        Item.status == 'Reclaimed',
        Item.reported_at.isnot(None),
        Item.reported_at <= end_datetime
    ).all()
    
    avg_return_days = 0.0
    if returned_with_dates:
        total_days = sum([
            (item.reported_at.date() - item.lost_found_date).days 
            for item in returned_with_dates 
            if item.lost_found_date
        ])
        avg_return_days = total_days / len(returned_with_dates) if returned_with_dates else 0.0
    
    # Popular locations (cumulative up to this date)
    location_counts = db.session.query(
        Item.location, func.count(Item.id)
    ).filter(
        Item.reported_at <= end_datetime,
        Item.is_active == True
    ).group_by(Item.location).order_by(func.count(Item.id).desc()).limit(10).all()
    
    popular_locations = json.dumps({loc: count for loc, count in location_counts})
    
    # Popular items (cumulative up to this date)
    item_counts = db.session.query(
        Item.item_name, func.count(Item.id)
    ).filter(
        Item.reported_at <= end_datetime,
        Item.is_active == True
    ).group_by(Item.item_name).order_by(func.count(Item.id).desc()).limit(10).all()
    
    popular_items = json.dumps({name: count for name, count in item_counts})
    
    # Peak hours (for items reported on this specific date)
    hour_counts = db.session.query(
        extract('hour', Item.reported_at).label('hour'),
        func.count(Item.id)
    ).filter(
        Item.reported_at >= start_datetime,
        Item.reported_at <= end_datetime
    ).group_by('hour').all()
    
    peak_hours = json.dumps({int(hour): count for hour, count in hour_counts if hour is not None})
    
    # Create analytics record
    analytics = Analytics(
        date=target_date,
        items_lost_count=items_lost,
        items_found_count=items_found,
        items_returned_count=items_returned,
        new_users_count=new_users,
        active_users_count=active_users,
        total_logins=total_logins,
        success_rate=success_rate,
        avg_return_time_days=avg_return_days,
        popular_locations=popular_locations,
        popular_items=popular_items,
        peak_hours=peak_hours
    )
    
    db.session.add(analytics)
    db.session.commit()
    
    return analytics


def get_analytics_summary(days=30):
    """Get analytics summary for the last N days"""
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    
    analytics = Analytics.query.filter(
        Analytics.date >= start_date,
        Analytics.date <= end_date
    ).order_by(Analytics.date.desc()).all()
    
    # If we don't have data for all days, generate it
    existing_dates = {a.date for a in analytics}
    current_date = start_date
    while current_date <= end_date:
        if current_date not in existing_dates:
            generate_daily_analytics(current_date)
        current_date += timedelta(days=1)
    
    # Fetch again after generation
    analytics = Analytics.query.filter(
        Analytics.date >= start_date,
        Analytics.date <= end_date
    ).order_by(Analytics.date.asc()).all()
    
    return analytics



# --- Models (Object-Relational Mapping) ---
# IMPORTANT: Item model is defined BEFORE User model to resolve
# "forward reference" issues for foreign_keys in User's relationships.
class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(10), default='Lost', nullable=False) # 'Lost', 'Found', 'Reclaimed'
    reported_at = db.Column(db.DateTime, default=datetime.now) # Timestamp when reported
    lost_found_date = db.Column(db.Date, nullable=False) # Date item was lost or found
    location = db.Column(db.String(200), nullable=False)
    image_filename = db.Column(db.String(255), nullable=True) # Stores filename for found items
    is_active = db.Column(db.Boolean, default=True) # For soft deletion/archiving

    # Foreign keys linking to the User model
    reported_by_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    deleted_by_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) # Can be null if not deleted
    deleted_at = db.Column(db.DateTime, nullable=True) # Timestamp when deleted/archived

    # OCR and verification fields (for found items with images)
    ocr_extracted_text = db.Column(db.Text, nullable=True) # Text extracted from image via OCR
    verification_score = db.Column(db.Float, nullable=True) # Cross-verification confidence (0.0-1.0)
    verification_details = db.Column(db.Text, nullable=True) # JSON string with detailed verification results

    def __repr__(self):
        # Defensive check for reporter existence before accessing .full_name
        reporter_name = self.reporter.full_name if self.reporter else 'Unknown'
        return f"<Item {self.item_name} - {self.status} by {reporter_name}>"


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    registration_no = db.Column(db.String(20), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    points = db.Column(db.Integer, default=0, nullable=False)
    level = db.Column(db.String(20), default='Bronze')
    avatar_filename = db.Column(db.String(255), nullable=True)
    role = db.Column(db.String(20), default='student') # 'student', 'admin', 'hod'
    
    # Privacy & GDPR Fields
    contact_visible = db.Column(db.Boolean, default=False, nullable=False)  # Hide contact until match confirmed
    email_public = db.Column(db.Boolean, default=False, nullable=False)  # Show email on profile
    phone = db.Column(db.String(15), nullable=True)  # Optional phone number
    phone_public = db.Column(db.Boolean, default=False, nullable=False)  # Show phone on profile
    
    # Multi-language preference
    preferred_language = db.Column(db.String(5), default='en', nullable=False)  # 'en', 'hi', 'gu'
    
    # Account tracking for GDPR compliance
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    last_login = db.Column(db.DateTime, nullable=True)
    data_consent = db.Column(db.Boolean, default=True, nullable=False)  # User consents to data processing

    # Relationships to Item model, explicitly defining foreign_keys
    items_reported = db.relationship(
        'Item',
        backref='reporter', # Creates item.reporter to access the User object
        lazy=True,
        foreign_keys=[Item.reported_by_id] # Explicitly tells SQLAlchemy which FK to use
    )
    items_deleted = db.relationship(
        'Item',
        backref='deleter', # Creates item.deleter to access the User object
        lazy=True,
        foreign_keys=[Item.deleted_by_id] # Explicitly tells SQLAlchemy which FK to use
    )

    def set_password(self, password):
        # Hash using passlib context (argon2 preferred)
        self.password_hash = pwd_context.hash(password)

    def check_password(self, password):
        # Try passlib verification first (supports argon2 and pbkdf2_sha256)
        try:
            verified = pwd_context.verify(password, self.password_hash)
        except Exception:
            # Fallback to werkzeug's check for legacy formats
            try:
                verified = check_password_hash(self.password_hash, password)
            except Exception:
                return False
        if verified:
            # Upgrade hash if the current hash uses a deprecated or weaker algorithm
            try:
                if pwd_context.needs_update(self.password_hash):
                    self.password_hash = pwd_context.hash(password)
                    db.session.add(self)
                    db.session.commit()
            except Exception:
                db.session.rollback()
        return verified

    def is_admin(self):
        return self.role == 'admin'

    def is_hod(self):
        return self.role == 'hod'

    def update_level(self):
        # Determine level from points
        p = self.points or 0
        if p >= 200:
            self.level = 'Platinum'
        elif p >= 50:
            self.level = 'Gold'
        elif p >= 10:
            self.level = 'Silver'
        else:
            self.level = 'Bronze'
    
    def get_contact_info(self, requester=None):
        """Return contact info based on privacy settings. GDPR-compliant."""
        # Admins/HODs can always see contacts
        if requester and (requester.is_admin() or requester.is_hod()):
            return {
                'email': self.email,
                'phone': self.phone,
                'visible': True
            }
        
        # If contact_visible is True, show to all
        if self.contact_visible:
            return {
                'email': self.email if self.email_public else self._mask_email(),
                'phone': self.phone if self.phone_public else self._mask_phone(),
                'visible': True
            }
        
        # Otherwise, mask everything
        return {
            'email': self._mask_email(),
            'phone': self._mask_phone(),
            'visible': False
        }
    
    def _mask_email(self):
        """Mask email for privacy (GDPR compliance)"""
        if not self.email:
            return 'Hidden'
        parts = self.email.split('@')
        if len(parts) != 2:
            return '***@***.***'
        username = parts[0]
        domain = parts[1]
        if len(username) <= 2:
            masked_user = '*' * len(username)
        else:
            masked_user = username[0] + '*' * (len(username) - 2) + username[-1]
        return f"{masked_user}@{domain}"
    
    def _mask_phone(self):
        """Mask phone for privacy (GDPR compliance)"""
        if not self.phone:
            return 'Hidden'
        if len(self.phone) <= 4:
            return '*' * len(self.phone)
        return self.phone[:2] + '*' * (len(self.phone) - 4) + self.phone[-2:]
    
    def export_user_data(self):
        """Export user data for GDPR compliance"""
        return {
            'registration_no': self.registration_no,
            'full_name': self.full_name,
            'email': self.email,
            'phone': self.phone,
            'points': self.points,
            'level': self.level,
            'role': self.role,
            'preferred_language': self.preferred_language,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'items_reported': [{'id': item.id, 'name': item.item_name, 'status': item.status} for item in self.items_reported if item.is_active]
        }

    def __repr__(self):
        return f"User('{self.registration_no}', '{self.full_name}', '{self.role}')"


class ReportLog(db.Model):
    """Audit log for item reports and deletions.

    Stores the user (by id and registration_no) who performed the action,
    the action type (reported_lost, reported_found, deleted), timestamp,
    optional linked item id and a short message/details.
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    registration_no = db.Column(db.String(20), nullable=True)
    action = db.Column(db.String(50), nullable=False)  # e.g., 'reported_lost', 'reported_found', 'deleted'
    item_id = db.Column(db.Integer, db.ForeignKey('item.id'), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    details = db.Column(db.Text, nullable=True)

    user = db.relationship('User', backref='logs', foreign_keys=[user_id])
    item = db.relationship('Item', backref='logs', foreign_keys=[item_id])
    

    def __repr__(self):
        # Helpful representation for debugging in the shell or logs
        return f"ReportLog(id={self.id}, reg_no={self.registration_no}, action={self.action}, item_id={self.item_id}, ts={self.timestamp})"


class AuthLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    registration_no = db.Column(db.String(20), nullable=True)
    action = db.Column(db.String(20), nullable=False)  # 'login' or 'logout'
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.now)
    ip_address = db.Column(db.String(45), nullable=True)

    user = db.relationship('User', backref='auth_logs', foreign_keys=[user_id])
    def __repr__(self):
        return f"AuthLog(id={self.id}, reg_no={self.registration_no}, action={self.action}, ts={self.timestamp})"
    @staticmethod
    def log_auth_action(user, action):
        ip = request.remote_addr if request else None
        al = AuthLog(user_id=user.id if user else None,
                    registration_no=(user.registration_no if user else None),
                    action=action,
                    timestamp= datetime.now(),
                    ip_address=ip)
        db.session.add(al)
        db.session.commit()
        print(f"DEBUG: AuthLog recorded: {action} by {al.registration_no} at {al.timestamp.isoformat()} from {ip}")


class Message(db.Model):
    """Simple per-item chat messages between users about an item."""
    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.Integer, db.ForeignKey('item.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)

    user = db.relationship('User', backref='messages', foreign_keys=[user_id])
    item = db.relationship('Item', backref='messages', foreign_keys=[item_id])

    def __repr__(self):
        return f"Message(id={self.id}, item_id={self.item_id}, user_id={self.user_id}, ts={self.timestamp})"

class Notification(db.Model):
    """Stores user specific alerts and notifications."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    link = db.Column(db.String(255), nullable=True) # Optional URL to redirect to
    is_read = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    type = db.Column(db.String(20), default='info') # 'info', 'match', 'message', 'success'

    user = db.relationship('User', backref='notifications', foreign_keys=[user_id])

    def __repr__(self):
        return f"Notification(id={self.id}, user_id={self.user_id}, read={self.is_read})"




class Analytics(db.Model):
    """Store daily analytics data for admin dashboard"""
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, unique=True)
    
    # Item statistics
    items_lost_count = db.Column(db.Integer, default=0)
    items_found_count = db.Column(db.Integer, default=0)
    items_returned_count = db.Column(db.Integer, default=0)
    
    # User engagement
    new_users_count = db.Column(db.Integer, default=0)
    active_users_count = db.Column(db.Integer, default=0)
    total_logins = db.Column(db.Integer, default=0)
    
    # Success metrics
    success_rate = db.Column(db.Float, default=0.0)  # Percentage of items returned
    avg_return_time_days = db.Column(db.Float, default=0.0)
    
    # Popular data (stored as JSON)
    popular_locations = db.Column(db.Text, nullable=True)  # JSON: {"location": count}
    popular_items = db.Column(db.Text, nullable=True)  # JSON: {"item_name": count}
    peak_hours = db.Column(db.Text, nullable=True)  # JSON: {"hour": count}
    
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<Analytics {self.date} - Lost: {self.items_lost_count}, Found: {self.items_found_count}>"


# --- WTForms (Web Forms) ---
# Forms are defined AFTER models, as they might reference models (e.g., for validation)
class RegistrationForm(FlaskForm):
    registration_no = StringField(_l('Registration No.'), validators=[
        DataRequired(), Length(min=5, max=20),
        Regexp('^[A-Za-z0-9]+$', message=_l("Registration number must contain only letters and digits."))
    ])
    full_name = StringField(_l('Full Name'), validators=[DataRequired(), Length(min=2, max=100)])
    email = StringField(_l('Email (Optional)'), validators=[Optional(), Length(max=120)])
    password = PasswordField(_l('Password'), validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField(_l('Confirm Password'), validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField(_l('Register'))

    # Custom validator to check if registration number already exists
    def validate_registration_no(self, registration_no):
        user = User.query.filter_by(registration_no=registration_no.data).first()
        if user:
            raise ValidationError(_l('That registration number is already taken. Please choose a different one.'))

    # Custom validator to check if email already exists
    def validate_email(self, email):
        if email.data and email.data.strip(): # Only validate if email is provided AND not empty/whitespace
            user = User.query.filter_by(email=email.data.strip()).first()
            if user:
                raise ValidationError(_l('That email address is already registered. Please use a different one or log in.'))


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
        DataRequired(),
        FileAllowed(ALLOWED_EXTENSIONS, _l('Images only! (png, jpg, jpeg, gif)'))
    ])
    submit = SubmitField(_l('Report Found Item'))
    # Optional confirmation for phone images; shown client-side when item is a phone
    phone_confirm = BooleanField(_l('I confirm this image shows the claimed item'))


class ProfilePhotoForm(FlaskForm):
    photo = FileField('Profile Photo', validators=[
        FileAllowed(ALLOWED_EXTENSIONS, 'Images only! (png, jpg, jpeg, gif)')
    ])
    submit = SubmitField('Upload')


class MessageForm(FlaskForm):
    content = TextAreaField(_l('Message'), validators=[DataRequired(), Length(min=1, max=1000)])
    submit = SubmitField(_l('Send'))


# Removed HistoryPasswordForm: history password protection was removed per user request.


# --- User Loader for Flask-Login ---
@login_manager.user_loader
def load_user(user_id):
    print(f"DEBUG: Flask-Login attempting to load user with ID: {user_id}")
    user = User.query.get(int(user_id))
    if user:
        print(f"DEBUG: User ID {user_id} loaded successfully: {user.full_name} ({user.role})")
    else:
        print(f"DEBUG: User ID {user_id} not found in database.")
    return user

# --- Routes (Application Logic) ---
@app.route("/health")
def health():
    # Basic health check that includes DB connection status
    db_status = "ok"
    try:
        # Simple query to check if DB is alive
        db.session.execute(db.text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return jsonify({
        "status": "live", 
        "ai_features": "enabled" if ENABLE_AI else "disabled",
        "database": db_status
    })

@app.route("/")
@app.route("/home")
def home():
    return render_template('index.html', title=_('Home'))

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        flash(_('You are already logged in.'), 'info')
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        try:
            email_data = form.email.data.strip() if form.email.data and form.email.data.strip() else None
            
            user = User(registration_no=form.registration_no.data,
                        full_name=form.full_name.data,
                        email=email_data,
                        role='student') # Explicitly set role to student for registrations
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            flash(_('Your account has been created! You are now able to log in'), 'success')
            print(f"DEBUG: New student registered: Reg No: {user.registration_no}, Email: {user.email}")
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            print(f"ERROR: Registration failed: {e}")
            import traceback
            traceback.print_exc()
            # Check if it's a unique constraint violation
            if 'UNIQUE constraint failed' in str(e) or 'unique' in str(e).lower():
                if 'registration_no' in str(e):
                    flash(_('That registration number is already taken. Please choose a different one.'), 'danger')
                elif 'email' in str(e):
                    flash(_('That email address is already registered. Please use a different one.'), 'danger')
                else:
                    flash(_('Registration failed: Duplicate entry detected.'), 'danger')
            else:
                flash(_('Registration failed due to a server error. Please try again. Error: %(error)s', error=str(e)), 'danger')
            return render_template('register.html', title=_('Register'), form=form)
    return render_template('register.html', title=_('Register'), form=form)

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        flash(_('You are already logged in.'), 'info')
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        attempted_reg_no = form.registration_no.data
        attempted_password = form.password.data

        print(f"\nDEBUG: Login attempt for Registration No: '{attempted_reg_no}'")

        user = User.query.filter_by(registration_no=attempted_reg_no).first()

        if user:
            print(f"DEBUG: User found in DB: Full Name='{user.full_name}', Role='{user.role}'")
            # For debugging, you can print the stored hash (DO NOT DO IN PRODUCTION)
            # print(f"DEBUG: Stored hash: {user.password_hash}")
            # print(f"DEBUG: Attempted password: {attempted_password}")

            if user.check_password(attempted_password):
                login_user(user)
                next_page = request.args.get('next')
                flash(_('Login successful!'), 'success')
                print(f"DEBUG: Login successful for user '{user.registration_no}' ({user.role}). Redirecting to {next_page or url_for('items')}")
                return redirect(next_page) if next_page else redirect(url_for('items'))
            else:
                print(f"DEBUG: Password mismatch for user '{user.registration_no}'.")
                flash(_('Login Unsuccessful. Please check registration number and password'), 'danger')
        else:
            print(f"DEBUG: No user found with Registration No: '{attempted_reg_no}'.")
            flash(_('Login Unsuccessful. Please check registration number and password'), 'danger')
    return render_template('login.html', title=_('Login'), form=form)

@app.route("/logout")
@login_required
def logout():
    print(f"DEBUG: User '{current_user.registration_no}' logging out.")
    logout_user()
    flash(_('You have been logged out.'), 'info')
    return redirect(url_for('home'))

@app.route("/report_lost", methods=['GET', 'POST'])
@login_required
def report_lost():
    form = ReportLostItemForm()
    if form.validate_on_submit():
        item = Item(item_name=form.item_name.data,
                    description=form.description.data,
                    lost_found_date=form.lost_date.data,
                    location=form.location.data,
                    status='Lost',
                    reporter=current_user)
        db.session.add(item)
        db.session.commit()
        # Log the report action
        log = ReportLog(user_id=current_user.id, registration_no=current_user.registration_no,
                        action='reported_lost', item_id=item.id,
                        details=f"Lost report: {item.item_name}")
        db.session.add(log)
        db.session.commit()
        flash(_('Your lost item report has been submitted!'), 'success')
        print(f"DEBUG: Lost item reported by {current_user.registration_no}: {item.item_name}")
        
        # After saving, check for matching active found reports
        matches = find_smart_matches(item)
        if matches:
            # Notify the reporter about matches
            create_notification(
                current_user.id,
                _('🤖 AI found %(count)d potential matches for your reported item: "%(name)s"!', count=len(matches), name=item.item_name),
                link=url_for('item_matches', item_id=item.id),
                type='match'
            )
            
            # Notify the finders that someone lost an item matching theirs
            for match in matches:
                matched_item = match['item']
                create_notification(
                    matched_item.reported_by_id,
                    _('Someone just reported a lost item ("%(name)s") that matches an item you found!', name=item.item_name),
                    link=url_for('item_detail', item_id=matched_item.id),
                    type='match'
                )

            flash(_('🤖 AI discovered %(count)d potential found reports that match your item!', count=len(matches)), 'info')
            return redirect(url_for('item_matches', item_id=item.id))

        return redirect(url_for('items'))
    return render_template('report_item_modern.html', title=_('Report Lost Item'), form=form, item_type='Lost', protected_keywords=PROTECTED_KEYWORDS)

@app.route("/report_found", methods=['GET', 'POST'])
@login_required
def report_found():
    print(f"DEBUG: Entering report_found. Method: {request.method}")
    form = ReportFoundItemForm()
    if request.method == 'POST' and not form.validate():
        print(f"DEBUG: Form validation failed. Errors: {form.errors}")
        
    if form.validate_on_submit():
        print(f"DEBUG: Form validated. Item: {form.item_name.data}")
        if 'image' not in request.files:
            flash(_('No file part'), 'danger')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash(_('No selected file'), 'danger')
            return redirect(request.url)

        # Server-side check: if item appears to be a protected/valuable type, ensure user ticked confirmation
        def looks_like_protected(name):
            if not name: return False
            ln = name.lower()
            return any(k in ln for k in PROTECTED_KEYWORDS)

        if looks_like_protected(form.item_name.data) and not form.phone_confirm.data:
            print(f"DEBUG: Protected item detected ('{form.item_name.data}') but confirmation missing. Re-rendering form.")
            flash(_('Protected/Valuable item detected. Please tick the confirmation box and re-upload your photo to proceed.'), 'warning')
            # Fall through to render_template instead of redirecting to avoid losing the uploaded data
            return render_template('report_item_modern.html', title=_('Report Found Item'), form=form, item_type='Found', protected_keywords=PROTECTED_KEYWORDS)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = app.config['UPLOAD_FOLDER']

            print(f"\nDEBUG: Calculated UPLOAD_FOLDER (absolute): {upload_path}")

            if not os.path.exists(upload_path):
                print(f"DEBUG: Directory '{upload_path}' does not exist. Attempting to create...")
                try:
                    os.makedirs(upload_path)
                    print(f"DEBUG: Successfully created upload directory: {upload_path}")
                except OSError as e:
                    print(f"ERROR: Failed to create directory '{upload_path}': {e}")
                    flash(_("Error creating upload directory: %(error)s. Check server permissions.", error=str(e)), 'danger')
                    return redirect(request.url)
            else:
                print(f"DEBUG: Directory '{upload_path}' already exists.")

            if not os.access(upload_path, os.W_OK):
                print(f"ERROR: Directory '{upload_path}' is not writable by the current user.")
                flash(_("Upload directory is not writable. Check server permissions for '%(path)s'.", path=upload_path), 'danger')
                return redirect(request.url)
            else:
                print(f"DEBUG: Directory '{upload_path}' is writable.")

            file_path = os.path.join(upload_path, filename)
            
            print(f"DEBUG: Calculated full file_path for saving: {file_path}")

            try:
                # Save uploaded image file to the configured upload folder.
                # This is an IO operation — it can fail if disk permissions or space are insufficient.
                file.save(file_path)
                # Optional: if Pillow is installed, perform a quick heuristic check
                try:
                    from PIL import Image, ImageFilter
                    img = Image.open(file_path)
                    w, h = img.size
                    # If protected item (e.g., phone/laptop) and image is extremely wide (panorama), warn the user
                    # Relaxed from 1.5 to 2.5 to allow landscape photos of phones/laptops.
                    if looks_like_protected(form.item_name.data) and (w > h * 2.5):
                        # For strict items, reject extremely wide/odd images rather than just warn
                        if any(k in form.item_name.data.lower() for k in PROTECTED_STRICT):
                            # remove saved file
                            try:
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                                    print(f"DEBUG: Removed image due to unacceptable aspect ratio: {file_path}")
                            except Exception as e:
                                print(f"DEBUG: Failed to remove file after aspect rejection: {e}")
                            flash(_('The uploaded image is too wide/distorted for this protected item. Please upload a clear, central photo of the claimed item and try again.'), 'warning')
                            return render_template('report_item_modern.html', title=_('Report Found Item'), form=form, item_type='Found', protected_keywords=PROTECTED_KEYWORDS)
                        else:
                            flash(_('Uploaded image seems unusually wide. Please upload a clear photo (portrait or closer framing).'), 'warning')

                    # Additional lightweight check for clarity: compute edge density for strict items when model not available
                    if any(k in form.item_name.data.lower() for k in PROTECTED_STRICT):
                        try:
                            # downscale to speed up the edge check
                            small = img.copy().resize((300, int(300 * (h/w)))) if w and h else img.copy()
                            edges = small.convert('L').filter(ImageFilter.FIND_EDGES)
                            # count bright pixels
                            arr = edges.point(lambda p: 1 if p > 30 else 0)
                            edge_count = sum(arr.getdata())
                            edge_density = edge_count / (arr.size[0] * arr.size[1])
                            print(f"DEBUG: Edge density for '{form.item_name.data}': {edge_density:.4f}")
                            # store for potential debugging; low density suggests blurry/blank
                            img.info['edge_density'] = edge_density
                        except Exception as e:
                            print(f"DEBUG: Edge check failed: {e}")
                    img.close()
                except Exception:
                    # Pillow not installed or image check failed; skip silently
                    pass
                print(f"DEBUG: File saved successfully to: {file_path}")

                # NEW: Pre-run object detection once to share results (Speed Optimization)
                yolo_detections = detect_with_yolo(file_path)

                # If the item name looks like a protected/valuable type, run server-side verification
                if looks_like_protected(form.item_name.data):
                    ok, info = verify_protected_image(form.item_name.data, file_path, yolo_dets=yolo_detections)
                    if not ok:
                        # Remove the saved file to avoid storing unverified images
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                print(f"DEBUG: Removed unverified uploaded file: {file_path}")
                        except Exception as e:
                            print(f"DEBUG: Failed to remove unverified file: {e}")

                        flash(_("The uploaded image does not appear to show the claimed item. Please upload a clear photo that shows the item (close-up or different angle) and try again. Reason: %(reason)s", reason=str(info)), 'warning')
                        print(f"DEBUG: Image verification failed for item '{form.item_name.data}'. Info: {info}")
                        return redirect(request.url)
                    else:
                        print(f"DEBUG: Image verification passed for item '{form.item_name.data}': {info}")
                
                # NEW: Extract text from image using OCR
                ocr_text = ""
                try:
                    success, ocr_text, ocr_details = extract_text_from_image(file_path)
                    if success and ocr_text:
                        print(f"DEBUG: OCR extracted text: {ocr_text[:100]}...")
                    else:
                        print("DEBUG: No text extracted from image")
                except Exception as e:
                    print(f"DEBUG: OCR extraction error: {e}")
                    ocr_text = ""
                
                # NEW: Cross-verify item name against image analysis
                verification_score = 0.0
                verification_details_dict = {}
                try:
                    verified, confidence, details = cross_verify_item(
                        form.item_name.data, 
                        file_path, 
                        extracted_text=ocr_text,
                        yolo_dets=yolo_detections
                    )
                    verification_score = confidence
                    verification_details_dict = details
                    print(f"DEBUG: Cross-verification - Confidence: {confidence:.2f}, Verified: {verified}")
                    
                    # Reject only for clear mismatches on high-value items or extremely low quality images
                    is_strict = any(k in form.item_name.data.lower() for k in PROTECTED_STRICT)
                    
                    should_reject = False
                    if confidence < 0.05: # Essential "reality check" to prevent empty/unusable uploads
                        should_reject = True
                        print(f"DEBUG: Rejecting due to garbage-level confidence: {confidence:.2f}")
                    elif has_conflict and is_strict: # Clear mismatch on protected high-value item
                        should_reject = True
                        print(f"DEBUG: Rejecting due to strict category conflict: {conflict_msg}")
                    
                    if should_reject:
                        # Log but do NOT remove the file to avoid broken images.
                        print(f"DEBUG: Low confidence/conflict for item '{form.item_name.data}'. Confidence: {confidence:.2f}")
                        
                        # Show detailed error message
                        if conflict_msg and is_strict:
                            error_msg = _('Item name mismatch for protected item. %(conflict)s. Please ensure the item name accurately describes what is in the photo.', conflict=conflict_msg)
                        elif confidence < 0.05:
                            error_msg = _('The uploaded image is too unclear or doesn\'t appear to contain a recognizable object (AI confidence %(confidence)d%%). Please upload a clearer photo.', confidence=int(confidence*100))
                        else:
                            error_msg = _('We are having trouble identifying this item. Please ensure you have uploaded a clear photo and mentioned the correct item name.')
                        
                        flash(error_msg, 'danger')
                        return redirect(request.url)
                    
                    # Show verification feedback to user for successful submissions
                    if confidence >= 0.7:
                        flash(_('✅ High confidence match (%(confidence)d%%) - Item verified!', confidence=int(confidence*100)), 'success')
                    elif confidence >= 0.4:
                        flash(_('⚠️ Medium confidence (%(confidence)d%%) - Please verify the details are correct.', confidence=int(confidence*100)), 'warning')
                    elif confidence >= 0.2:
                        flash(_('⚠️ Low confidence match (%(confidence)d%%) - Please double-check item name and image match.', confidence=int(confidence*100)), 'warning')
                        
                except Exception as e:
                    print(f"DEBUG: Cross-verification error: {e}")
            except Exception as e:
                print(f"ERROR: Failed to save file '{filename}' to '{file_path}': {e}")
                flash(_("Error saving image: %(error)s. Check server disk space or permissions.", error=str(e)), 'danger')
                return redirect(request.url)

            # Store verification details as JSON
            import json
            verification_json = json.dumps(verification_details_dict) if verification_details_dict else None
            
            print(f"DEBUG: Saving item to DB: {form.item_name.data}")

            item = Item(item_name=form.item_name.data,
                        description=form.description.data,
                        lost_found_date=form.found_date.data,
                        location=form.location.data,
                        status='Found',
                        image_filename=filename,
                        ocr_extracted_text=ocr_text,
                        verification_score=verification_score,
                        verification_details=verification_json,
                        reporter=current_user)
            db.session.add(item)
            print("DEBUG: Item added to session. Committing...")
            db.session.commit()
            print(f"DEBUG: Item committed. ID: {item.id}")
            # Log the found report
            log = ReportLog(user_id=current_user.id, registration_no=current_user.registration_no,
                            action='reported_found', item_id=item.id,
                            details=f"Found report: {item.item_name} (Image: {filename})")
            db.session.add(log)
            db.session.commit()
            # After saving the found report, check for matching active lost reports
            matches = find_smart_matches(item)
            points_awarded = 0
            if matches:
                # Award points for potential matches
                points_awarded = len(matches)
                current_user.points = (current_user.points or 0) + points_awarded
                current_user.update_level()
                db.session.commit()
                # Log the point award
                plog = ReportLog(user_id=current_user.id, registration_no=current_user.registration_no,
                                action='points_awarded', item_id=item.id,
                                details=f"Awarded {points_awarded} points for AI smart-matching {len(matches)} lost report(s)")
                db.session.add(plog)
                
                # Notify the finder
                create_notification(
                    current_user.id,
                    _('🤖 Great job! AI found %(count)d lost reports matching your found item. You earned %(points)d points!', count=len(matches), points=points_awarded),
                    link=url_for('item_matches', item_id=item.id),
                    type='success'
                )

                # Notify the owners of lost items
                for match in matches:
                    matched_item = match['item']
                    create_notification(
                        matched_item.reported_by_id,
                        _('Someone found an item that might be your "%(name)s"! Check it out.', name=matched_item.item_name),
                        link=url_for('item_detail', item_id=item.id),
                        type='match'
                    )
                db.session.commit()

            if points_awarded:
                msg = _("Your found item report has been submitted! You earned %(points)d points!", points=points_awarded)
            else:
                msg = _("Your found item report has been submitted!")
            flash(msg, 'success')
            print(f"DEBUG: Found item reported by {current_user.registration_no}: {item.item_name}. Points awarded: {points_awarded}")

            if matches:
                flash(_('🤖 AI discovered %(count)d potential lost reports that match your item!', count=len(matches)), 'info')
                return redirect(url_for('item_matches', item_id=item.id))
            
            return redirect(url_for('items'))
        else:
            flash(_("Invalid file type. Allowed: %(extensions)s", extensions=', '.join(app.config['ALLOWED_EXTENSIONS'])), 'danger')
    return render_template('report_item_modern.html', title=_('Report Found Item'), form=form, item_type='Found', protected_keywords=PROTECTED_KEYWORDS)

@app.route("/items")
@login_required
def items():
    # Support optional search via ?q=term
    q = request.args.get('q', '').strip()
    # Support optional filter via ?filter=lost|found
    f = request.args.get('filter', '').strip().lower()
    
    query = Item.query.filter_by(is_active=True)
    
    if q:
        # Filter by name, description or location (case-insensitive)
        from sqlalchemy import or_
        query = query.filter(
            or_(
                Item.item_name.ilike(f"%{q}%"),
                Item.description.ilike(f"%{q}%"),
                Item.location.ilike(f"%{q}%")
            )
        )
    
    if f == 'lost':
        query = query.filter_by(status='Lost')
    elif f == 'found':
        query = query.filter_by(status='Found')

    active_items = query.order_by(Item.reported_at.desc()).all()
    print(f"DEBUG: Displaying {len(active_items)} active items (filter='{f}', search='{q}') for {current_user.registration_no}")
    return render_template('item_list.html', title=_('Lost & Found Items'), items=active_items)

@app.route("/item/<int:item_id>", methods=['GET', 'POST'])
@login_required
def item_detail(item_id):
    item = db.session.get(Item, item_id)
    if item is None:
        # Item missing return 404 to the user
        print(f"DEBUG: Item with ID {item_id} not found for detail view.")
        return render_template('404.html', title=_('Item Not Found')), 404

    form = MessageForm()
    if form.validate_on_submit():
        # Create a new message for this item
        msg = Message(item_id=item.id, user_id=current_user.id, content=form.content.data)
        db.session.add(msg)
        db.session.commit()
        # Log message creation in ReportLog for audit (optional)
        mlog = ReportLog(user_id=current_user.id, registration_no=current_user.registration_no,
                         action='message_posted', item_id=item.id,
                         details=(form.content.data[:200] + '...' if len(form.content.data) > 200 else form.content.data))
        db.session.add(mlog)
        db.session.commit()
        flash(_('Message posted.'), 'success')
        
        # Notify the other party
        # The other party is either the reporter of the item or anyone who has messaged about this item
        # Simplified: notify the reporter if the messenger is someone else, or notify everyone else who messaged
        if current_user.id != item.reported_by_id:
            # Notifier is a viewer, notify the reporter
            create_notification(
                item.reported_by_id,
                _('New message from %(user)s about your item: "%(name)s"', user=current_user.full_name, name=item.item_name),
                link=url_for('item_detail', item_id=item.id),
                type='message'
            )
        else:
            # Notifier is the reporter, notify everyone who messaged
            messengers = db.session.query(Message.user_id).filter_by(item_id=item.id).distinct().all()
            for (uid,) in messengers:
                if uid != current_user.id:
                    create_notification(
                        uid,
                        _('Reporter messaged about "%(name)s"', name=item.item_name),
                        link=url_for('item_detail', item_id=item.id),
                        type='message'
                    )
        
        return redirect(url_for('item_detail', item_id=item.id))

    # Load messages for this item (most recent last)
    messages = Message.query.filter_by(item_id=item.id).order_by(Message.timestamp.asc()).all()
    print(f"DEBUG: Displaying details for item ID {item_id}: {item.item_name} with {len(messages)} messages")
    return render_template('item_detail_modern.html', title=_('Item Details'), item=item, form=form, messages=messages)


@app.route("/item/<int:item_id>/delete", methods=['POST'])
@login_required
def delete_item(item_id):
    # Server-side authorization: only admin or HOD can delete items.
    if not (current_user.is_admin() or current_user.is_hod()):
        # Log the unauthorized attempt and return 403 Forbidden
        print(f"DEBUG: Unauthorized delete attempt by {current_user.registration_no} (Role: {current_user.role}) on item ID {item_id}.")
        abort(403)

    item = db.session.get(Item, item_id)
    if item is None:
        flash(_('Item not found.'), 'danger')
        print(f"DEBUG: Delete attempt for non-existent item ID {item_id}.")
        return redirect(url_for('items'))

    if item.is_active:
        item.is_active = False
        item.deleted_by = current_user
        item.deleted_at = datetime.utcnow()
        db.session.commit()
        # Log the deletion action in the audit log
        log = ReportLog(user_id=current_user.id, registration_no=current_user.registration_no,
                        action='deleted', item_id=item.id,
                        details=f"Item moved to history: {item.item_name}")
        db.session.add(log)
        db.session.commit()
        flash(_('Item has been successfully moved to history.'), 'success')
        print(f"DEBUG: Item '{item.item_name}' (ID: {item_id}) soft-deleted by {current_user.registration_no}.")
    else:
        flash(_('Item is already in history.'), 'info')
        print(f"DEBUG: Item '{item.item_name}' (ID: {item_id}) already inactive, no action taken.")
    return redirect(url_for('items'))


@app.context_processor
def utility_processor():
    # Expose a helper to templates to check delete permission
    def can_delete_item(user):
        if not user:
            return False
        return user.is_admin() or user.is_hod()
    return dict(can_delete_item=can_delete_item, current_year=datetime.now().year)


@app.errorhandler(403)
def forbidden(error):
    # Provide a friendly forbidden page and log the event
    print(f"DEBUG: 403 Forbidden - {request.remote_addr} attempted an unauthorized action. Message: {error}")
    return render_template('403.html', title=_('Forbidden')), 403

@app.errorhandler(500)
def internal_server_error(error):
    # Log the full error with stack trace
    print(f"ERROR: 500 Internal Server Error occurred")
    print(f"ERROR: Request URL: {request.url}")
    print(f"ERROR: Request Method: {request.method}")
    print(f"ERROR: Error: {error}")
    import traceback
    traceback.print_exc()
    
    # Rollback any pending database transactions
    try:
        db.session.rollback()
    except Exception:
        pass
    
    # Return a user-friendly error page
    return render_template('500.html', title=_('Server Error')), 500

# =====================================================
# Admin Panel Routes
# =====================================================

@app.route("/admin")
@app.route("/admin/dashboard")
@login_required
def admin_dashboard():
    """Admin dashboard with user management, transactions, and analytics"""
    # Only admin or hod can access
    if not (current_user.is_admin() or current_user.is_hod()):
        flash(_('You do not have permission to access the admin panel.'), 'danger')
        print(f"DEBUG: User {current_user.registration_no} (Role: {current_user.role}) attempted unauthorized access to admin panel.")
        return redirect(url_for('items'))
    
    # Get all users
    users = User.query.order_by(User.id.desc()).all()
    
    # Get all transactions (report logs)
    transactions = ReportLog.query.order_by(ReportLog.timestamp.desc()).limit(100).all()
    
    # Get all items
    items = Item.query.order_by(Item.reported_at.desc()).all()
    
    # Calculate statistics
    stats = {
        'total_users': User.query.count(),
        'active_items': Item.query.filter_by(is_active=True).count(),
        'total_transactions': ReportLog.query.count(),
        'archived_items': Item.query.filter_by(is_active=False).count()
    }
    
    # Calculate analytics
    from datetime import timedelta
    from sqlalchemy import func
    
    students_count = User.query.filter_by(role='student').count()
    admins_count = User.query.filter_by(role='admin').count()
    hods_count = User.query.filter_by(role='hod').count()
    
    # Average points
    avg_points_result = db.session.query(func.avg(User.points)).scalar()
    avg_points = avg_points_result if avg_points_result else 0
    
    # Lost vs Found
    lost_items = Item.query.filter_by(status='Lost', is_active=True).count()
    found_items = Item.query.filter_by(status='Found', is_active=True).count()
    
    # Items this week
    week_ago = datetime.utcnow() - timedelta(days=7)
    items_this_week = Item.query.filter(Item.reported_at >= week_ago).count()
    
    # Match rate (simplified - items with messages as proxy for matches)
    total_active = Item.query.filter_by(is_active=True).count()
    items_with_messages = db.session.query(Item.id).join(Message).filter(Item.is_active == True).distinct().count()
    match_rate = int((items_with_messages / total_active * 100)) if total_active > 0 else 0
    
    # Top contributors
    top_contributors = db.session.query(
        User,
        func.count(Item.id).label('items_count')
    ).join(Item, Item.reported_by_id == User.id).group_by(User.id).order_by(
        func.count(Item.id).desc()
    ).limit(5).all()
    
    top_contributors_list = []
    for user, count in top_contributors:
        top_contributors_list.append({
            'full_name': user.full_name,
            'registration_no': user.registration_no,
            'items_count': count,
            'points': user.points or 0
        })
    
    analytics = {
        'students_count': students_count,
        'admins_count': admins_count,
        'hods_count': hods_count,
        'avg_points': avg_points,
        'lost_items': lost_items,
        'found_items': found_items,
        'match_rate': match_rate,
        'items_this_week': items_this_week,
        'top_contributors': top_contributors_list
    }
    
    print(f"DEBUG: Admin dashboard accessed by {current_user.registration_no}. Showing {len(users)} users, {len(transactions)} transactions.")
    
    return render_template('admin_dashboard.html', 
                         title=_('Admin Dashboard'),
                         users=users,
                         transactions=transactions,
                         items=items,
                         stats=stats,
                         analytics=analytics)


@app.route("/admin/user/<int:user_id>")
@login_required
def admin_user_detail(user_id):
    """View detailed information about a specific user"""
    if not (current_user.is_admin() or current_user.is_hod()):
        abort(403)
    
    user = User.query.get_or_404(user_id)
    
    # Get user's items
    user_items = Item.query.filter_by(reported_by_id=user.id).order_by(Item.reported_at.desc()).all()
    
    # Get user's logs
    user_logs = ReportLog.query.filter_by(user_id=user.id).order_by(ReportLog.timestamp.desc()).limit(50).all()
    
    # Get user's auth logs
    user_auth_logs = AuthLog.query.filter_by(user_id=user.id).order_by(AuthLog.timestamp.desc()).limit(20).all()
    
    return render_template('admin_user_detail.html',
                         title=_('User: %(name)s', name=user.full_name),
                         user=user,
                         items=user_items,
                         logs=user_logs,
                         auth_logs=user_auth_logs)


@app.route("/admin/user/<int:user_id>/role", methods=['POST'])
@login_required
def admin_change_user_role(user_id):
    """Change a user's role (admin only)"""
    if not current_user.is_admin():
        return jsonify({'success': False, 'message': 'Only admins can change user roles'}), 403
    
    user = User.query.get_or_404(user_id)
    
    data = request.get_json()
    new_role = data.get('role', '').lower()
    
    if new_role not in ['student', 'admin', 'hod']:
        return jsonify({'success': False, 'message': 'Invalid role'}), 400
    
    old_role = user.role
    user.role = new_role
    db.session.commit()
    
    # Log the role change
    log = ReportLog(
        user_id=current_user.id,
        registration_no=current_user.registration_no,
        action='role_changed',
        details=f"Changed {user.registration_no}'s role from {old_role} to {new_role}"
    )
    db.session.add(log)
    db.session.commit()
    
    print(f"DEBUG: Admin {current_user.registration_no} changed user {user.registration_no} role from {old_role} to {new_role}")
    
    return jsonify({'success': True, 'message': f'Role updated to {new_role}'})


@app.route("/history", methods=['GET', 'POST'])
@login_required
def item_history():
    # Only admin or hod can attempt to view history
    if not (current_user.is_admin() or current_user.is_hod()):
        flash(_('You do not have permission to view item history.'), 'danger')
        print(f"DEBUG: User {current_user.registration_no} (Role: {current_user.role}) attempted unauthorized access to history.")
        return redirect(url_for('items'))

    # Only users with admin or hod roles can view history — render it directly.
    all_items = Item.query.order_by(Item.reported_at.desc()).all()
    print(f"DEBUG: Displaying {len(all_items)} items (including inactive) for {current_user.registration_no}.")
    return render_template('item_history.html', title=_('Item History'), items=all_items)

# DEBUGGING ROUTE - Remove in production
@app.route("/debug/users")
def debug_users():
    users = User.query.all()
    user_info = []
    for user in users:
        user_info.append({
            'id': user.id,
            'reg_no': user.registration_no,
            'name': user.full_name,
            'email': user.email,
            'role': user.role,
            'avatar': user.avatar_filename,
            'has_password': True if user.password_hash else False
        })
    print(f"DEBUG: /debug/users requested. Found {len(users)} users.")
    # Return as JSON for easy inspection in browser
    return user_info # Flask will automatically jsonify this list of dicts


@app.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = ProfilePhotoForm()
    if form.validate_on_submit():
        file = request.files.get('photo')
        if not file or file.filename == '':
            flash(_('No file selected.'), 'warning')
            return redirect(url_for('edit_profile'))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # prefix filename with user id and timestamp to avoid collisions
            name, ext = os.path.splitext(filename)
            timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
            save_name = f"user_{current_user.id}_{timestamp}{ext}"
            upload_path = app.config['UPLOAD_FOLDER']
            if not os.path.exists(upload_path):
                try:
                    os.makedirs(upload_path)
                except OSError as e:
                    flash(_('Unable to create upload directory: %(error)s', error=str(e)), 'danger')
                    return redirect(url_for('edit_profile'))

            file_path = os.path.join(upload_path, save_name)
            try:
                file.save(file_path)
            except Exception as e:
                flash(_('Failed to save file: %(error)s', error=str(e)), 'danger')
                return redirect(url_for('edit_profile'))

            # Delete previous avatar file if present and not a shared placeholder
            if current_user.avatar_filename:
                try:
                    prev_path = os.path.join(upload_path, current_user.avatar_filename)
                    if os.path.exists(prev_path):
                        os.remove(prev_path)
                except Exception:
                    # Ignore failures to remove previous avatar
                    pass

            current_user.avatar_filename = save_name
            db.session.commit()
            flash(_('Profile photo updated.'), 'success')
            return redirect(url_for('profile'))
        else:
            flash(_('Invalid file type.'), 'danger')
            return redirect(url_for('edit_profile'))

    return render_template('edit_profile.html', title=_('Edit Profile'), form=form, user=current_user)


@app.route("/debug/logs")
def debug_logs():
    # Returns the 200 most recent report logs for quick inspection
    logs = ReportLog.query.order_by(ReportLog.timestamp.desc()).limit(200).all()
    out = []
    for l in logs:
        out.append({
            'id': l.id,
            'reg_no': l.registration_no,
            'user_id': l.user_id,
            'action': l.action,
            'item_id': l.item_id,
            'timestamp': l.timestamp.isoformat(),
            'details': l.details,
        })
    print(f"DEBUG: /debug/logs requested. Found {len(out)} log entries.")
    return out


@app.route("/levels")
@login_required
def levels():
    # Show current user's points and level
    user = current_user
    # Ensure level is up-to-date
    user.update_level()
    db.session.commit()
    return render_template('levels.html', title=_('Your Level'), user=user)


@app.route("/leaderboard")
@login_required
def leaderboard():
    # Show top users by points
    top_users = User.query.order_by(User.points.desc()).limit(50).all()
    return render_template('leaderboard.html', title=_('Leaderboard'), users=top_users)


@app.route("/profile")
@login_required
def profile():
    user = current_user
    # Ensure level reflects current points
    user.update_level()
    # Recent reports by the user (last 10)
    recent_reports = Item.query.filter_by(reported_by_id=user.id).order_by(Item.reported_at.desc()).limit(10).all()
    # Compute progress toward next level
    p = user.points or 0
    # Level thresholds (lower, upper). upper None means max
    thresholds = {
        'Bronze': (0, 10),
        'Silver': (10, 50),
        'Gold': (50, 200),
        'Platinum': (200, None)
    }
    lower, upper = thresholds.get(user.level, (0, 10))
    if upper is None:
        progress_pct = 100
        points_to_next = None
        next_level = None
    else:
        span = upper - lower
        progress_pct = int(max(0, min(100, ((p - lower) / span) * 100))) if span > 0 else 0
        points_to_next = max(0, upper - p)
        # find next level name
        level_order = ['Bronze', 'Silver', 'Gold', 'Platinum']
        try:
            idx = level_order.index(user.level)
            next_level = level_order[idx+1] if idx+1 < len(level_order) else None
        except ValueError:
            next_level = None

    return render_template('profile.html', title=_('Your Profile'), user=user, reports=recent_reports,
                           progress_pct=progress_pct, points_to_next=points_to_next, next_level=next_level)


# =====================================================
# API Endpoints for AJAX Requests
# =====================================================

@app.route('/api/extract_text', methods=['POST'])
@login_required
def api_extract_text():
    """
    API endpoint to extract text from uploaded image using OCR.
    Returns JSON with extracted text and item name suggestions.
    """
    import json
    import tempfile
    
    if 'image' not in request.files:
        return json.dumps({'success': False, 'message': 'No image file provided'}), 400, {'ContentType': 'application/json'}
    
    file = request.files['image']
    if file.filename == '':
        return json.dumps({'success': False, 'message': 'No file selected'}), 400, {'ContentType': 'application/json'}
    
    if not allowed_file(file.filename):
        return json.dumps({'success': False, 'message': 'Invalid file type. Please upload PNG, JPG, or GIF'}), 400, {'ContentType': 'application/json'}
    
    try:
        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_path = temp_file.name
            file.save(temp_path)
        
        # Extract text using OCR
        success, extracted_text, ocr_details = extract_text_from_image(temp_path)
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        if not success:
            return json.dumps({
                'success': False, 
                'message': 'OCR extraction failed. Image may not contain readable text.'
            }),200, {'ContentType': 'application/json'}
        
        # Generate item name suggestions from extracted text
        suggestions = suggest_item_name(extracted_text)
        
        return json.dumps({
            'success': True,
            'extracted_text': extracted_text,
            'suggestions': suggestions,
            'ocr_details': ocr_details
        }), 200, {'ContentType': 'application/json'}
        
    except Exception as e:
        print(f"ERROR: API extract_text failed: {e}")
        return json.dumps({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500, {'ContentType': 'application/json'}



@app.route('/item/<int:item_id>/matches')
@login_required
def item_matches(item_id):
    item = Item.query.get_or_404(item_id)
    matches = find_smart_matches(item)
    return render_template('item_matches.html', item=item, matches=matches)


# --- Analytics Dashboard Routes (Admin/HOD Only) ---
@app.route('/analytics')
@login_required
def analytics_dashboard():
    """Analytics dashboard for admins and HODs"""
    if not (current_user.is_admin() or current_user.is_hod()):
        flash(_('Access denied. Admin or HOD privileges required.'), 'danger')
        return redirect(url_for('index'))
    
    # Get analytics for the last 30 days
    analytics_data = get_analytics_summary(days=30)
    
    # Calculate overall statistics
    total_items_lost = sum(a.items_lost_count for a in analytics_data)
    total_items_found = sum(a.items_found_count for a in analytics_data)
    total_items_returned = sum(a.items_returned_count for a in analytics_data)
    
    # Get current analytics
    today_analytics = generate_daily_analytics(date.today())
    
    # Get all-time statistics
    total_users = User.query.count()
    total_items = Item.query.filter_by(is_active=True).count()
    items_lost_active = Item.query.filter_by(status='Lost', is_active=True).count()
    items_found_active = Item.query.filter_by(status='Found', is_active=True).count()
    
    # Popular locations and items (from latest analytics)
    popular_locations = json.loads(today_analytics.popular_locations) if today_analytics.popular_locations else {}
    popular_items = json.loads(today_analytics.popular_items) if today_analytics.popular_items else {}
    peak_hours = json.loads(today_analytics.peak_hours) if today_analytics.peak_hours else {}
    
    return render_template('analytics_dashboard.html',
                         title=_('Analytics Dashboard'),
                         analytics_data=analytics_data,
                         total_items_lost=total_items_lost,
                         total_items_found=total_items_found,
                         total_items_returned=total_items_returned,
                         total_users=total_users,
                         total_items=total_items,
                         items_lost_active=items_lost_active,
                         items_found_active=items_found_active,
                         popular_locations=popular_locations,
                         popular_items=popular_items,
                         peak_hours=peak_hours,
                         success_rate=today_analytics.success_rate,
                         avg_return_days=today_analytics.avg_return_time_days)


# --- Export Reports Routes (Admin/HOD Only) ---
@app.route('/export/items/<format>')
@login_required
def export_items(format):
    """Export items data to CSV or PDF (Admin/HOD only)"""
    if not (current_user.is_admin() or current_user.is_hod()):
        flash(_('Access denied. Admin or HOD privileges required.'), 'danger')
        return redirect(url_for('index'))
    
    # Get filter parameters
    status_filter = request.args.get('status', 'all')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Build query
    query = Item.query.filter_by(is_active=True)
    
    if status_filter != 'all':
        query = query.filter_by(status=status_filter)
    
    if start_date:
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            query = query.filter(Item.reported_at >= start_dt)
        except ValueError:
            pass
    
    if end_date:
        try:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            end_dt = end_dt.replace(hour=23, minute=59, second=59)
            query = query.filter(Item.reported_at <= end_dt)
        except ValueError:
            pass
    
    items = query.order_by(Item.reported_at.desc()).all()
    
    if format == 'csv':
        return export_items_csv(items)
    elif format == 'pdf':
        if not REPORTLAB_AVAILABLE:
            flash(_('PDF export is not available. Please install reportlab.'), 'warning')
            return redirect(url_for('analytics_dashboard'))
        return export_items_pdf(items)
    else:
        flash(_('Invalid export format'), 'danger')
        return redirect(url_for('analytics_dashboard'))


def export_items_csv(items):
    """Generate CSV export of items"""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['ID', 'Item Name', 'Description', 'Status', 'Location', 
                     'Date Lost/Found', 'Reported At', 'Reported By', 'Reporter Email'])
    
    # Write data
    for item in items:
        reporter_name = item.reporter.full_name if item.reporter else 'Unknown'
        reporter_email = item.reporter.email if item.reporter else 'N/A'
        
        writer.writerow([
            item.id,
            item.item_name,
            item.description,
            item.status,
            item.location,
            item.lost_found_date.strftime('%Y-%m-%d') if item.lost_found_date else 'N/A',
            item.reported_at.strftime('%Y-%m-%d %H:%M:%S') if item.reported_at else 'N/A',
            reporter_name,
            reporter_email
        ])
    
    # Create response
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = f'attachment; filename=items_export_{date.today()}.csv'
    response.headers['Content-Type'] = 'text/csv'
    
    return response


def export_items_pdf(items):
    """Generate PDF export of items"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    
    # Add title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=1  # Center
    )
    elements.append(Paragraph('Lost & Found Portal - Items Report', title_style))
    elements.append(Paragraph(f'Generated on: {date.today().strftime("%B %d, %Y")}', styles['Normal']))
    elements.append(Spacer(1, 0.5*inch))
    
    # Prepare table data
    table_data = [['ID', 'Item Name', 'Status', 'Location', 'Date', 'Reported By']]
    
    for item in items:
        reporter_name = item.reporter.full_name if item.reporter else 'Unknown'
        date_str = item.lost_found_date.strftime('%Y-%m-%d') if item.lost_found_date else 'N/A'
        
        table_data.append([
            str(item.id),
            item.item_name[:30],  # Truncate long names
            item.status,
            item.location[:25],  # Truncate long locations
            date_str,
            reporter_name[:20]  # Truncate long names
        ])
    
    # Create table
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    elements.append(table)
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    
    return send_file(buffer, as_attachment=True, 
                    download_name=f'items_report_{date.today()}.pdf',
                    mimetype='application/pdf')


@app.route('/export/analytics/<format>')
@login_required
def export_analytics(format):
    """Export analytics data to CSV or PDF (Admin/HOD only)"""
    if not (current_user.is_admin() or current_user.is_hod()):
        flash(_('Access denied. Admin or HOD privileges required.'), 'danger')
        return redirect(url_for('index'))
    
    days = int(request.args.get('days', 30))
    analytics_data = get_analytics_summary(days=days)
    
    if format == 'csv':
        return export_analytics_csv(analytics_data)
    elif format == 'pdf':
        if not REPORTLAB_AVAILABLE:
            flash(_('PDF export is not available.'), 'warning')
            return redirect(url_for('analytics_dashboard'))
        return export_analytics_pdf(analytics_data)
    else:
        flash(_('Invalid export format'), 'danger')
        return redirect(url_for('analytics_dashboard'))


def export_analytics_csv(analytics_data):
    """Generate CSV export of analytics"""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Date', 'Items Lost', 'Items Found', 'Items Returned', 
                     'New Users', 'Active Users', 'Logins', 'Success Rate %', 'Avg Return Days'])
    
    # Write data
    for a in analytics_data:
        writer.writerow([
            a.date.strftime('%Y-%m-%d'),
            a.items_lost_count,
            a.items_found_count,
            a.items_returned_count,
            a.new_users_count,
            a.active_users_count,
            a.total_logins,
            f'{a.success_rate:.2f}',
            f'{a.avg_return_time_days:.1f}'
        ])
    
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = f'attachment; filename=analytics_export_{date.today()}.csv'
    response.headers['Content-Type'] = 'text/csv'
    
    return response


def export_analytics_pdf(analytics_data):
    """Generate PDF export of analytics"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    
    # Add title
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], 
                                fontSize=24, textColor=colors.HexColor('#2c3e50'),
                                spaceAfter=30, alignment=1)
    elements.append(Paragraph('Lost & Found Portal - Analytics Report', title_style))
    elements.append(Paragraph(f'Generated on: {date.today().strftime("%B %d, %Y")}', styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Summary statistics
    total_lost = sum(a.items_lost_count for a in analytics_data)
    total_found = sum(a.items_found_count for a in analytics_data)
    total_returned = sum(a.items_returned_count for a in analytics_data)
    avg_success = sum(a.success_rate for a in analytics_data) / len(analytics_data) if analytics_data else 0
    
    summary = f"""
    <b>Summary Statistics:</b><br/>
    Total Items Lost: {total_lost}<br/>
    Total Items Found: {total_found}<br/>
    Total Items Returned: {total_returned}<br/>
    Average Success Rate: {avg_success:.2f}%<br/>
    """
    elements.append(Paragraph(summary, styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Daily data table
    table_data = [['Date', 'Lost', 'Found', 'Returned', 'Success %']]
    for a in analytics_data:
        table_data.append([
            a.date.strftime('%Y-%m-%d'),
            str(a.items_lost_count),
            str(a.items_found_count),
            str(a.items_returned_count),
            f'{a.success_rate:.1f}%'
        ])
    
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    
    return send_file(buffer, as_attachment=True,
                    download_name=f'analytics_report_{date.today()}.pdf',
                    mimetype='application/pdf')


# --- Language Switcher Route ---
@app.route('/set_language/<lang>')
def set_language(lang):
    """Set user's preferred language"""
    if lang not in app.config.get('BABEL_SUPPORTED_LOCALES', ['en']):
        flash(_('Invalid language selection'), 'danger')
        return redirect(request.referrer or url_for('index'))
    
    # Update user preference if logged in
    if current_user.is_authenticated:
        current_user.preferred_language = lang
        db.session.commit()
        flash(_('Language updated successfully'), 'success')
    
    # Set in session
    from flask import session
    session['language'] = lang
    
    return redirect(request.referrer or url_for('index'))


# --- Privacy Settings Route ---
@app.route('/privacy/settings', methods=['GET', 'POST'])
@login_required
def privacy_settings():
    """User privacy settings (GDPR compliance)"""
    if request.method == 'POST':
        current_user.contact_visible = 'contact_visible' in request.form
        current_user.email_public = 'email_public' in request.form
        current_user.phone_public = 'phone_public' in request.form
        
        phone = request.form.get('phone', '').strip()
        if phone:
            current_user.phone = phone
        
        db.session.commit()
        flash(_('Privacy settings updated successfully'), 'success')
        return redirect(url_for('privacy_settings'))
    
    return render_template('privacy_settings.html', title=_('Privacy Settings'))


# --- GDPR Data Export Route ---
@app.route('/export/my_data')
@login_required
def export_my_data():
    """Export user's personal data (GDPR right to data portability)"""
    user_data = current_user.export_user_data()
    
    # Create JSON file
    output = io.BytesIO()
    output.write(json.dumps(user_data, indent=2).encode('utf-8'))
    output.seek(0)
    
    return send_file(output, as_attachment=True,
                    download_name=f'my_data_{current_user.registration_no}_{date.today()}.json',
                    mimetype='application/json')



def init_db():
    with app.app_context():
        # Ensure the upload directory exists at startup too, for robustness
        if not os.path.exists(UPLOAD_FOLDER):
            try:
                os.makedirs(UPLOAD_FOLDER)
                print(f"Startup DEBUG: Created upload directory: {UPLOAD_FOLDER}")
            except OSError as e:
                print(f"Startup ERROR: Failed to create upload directory {UPLOAD_FOLDER}: {e}")
                print("Please check file system permissions for your project folder.")

        db.create_all() # Creates database tables based on your models
        print("Startup DEBUG: Database tables verified/created.")

        # Create default admin and HOD users if they don't exist
        admin_exists = User.query.filter_by(registration_no='ADMIN001').first()
        if not admin_exists:
            admin_user = User(registration_no='ADMIN001', full_name='Admin User', email='admin@example.com', role='admin')
            admin_user.set_password('adminpassword') # Use set_password method
            db.session.add(admin_user)
            db.session.commit()
            print("Startup DEBUG: Default admin user created: ADMIN001 / adminpassword")
        else:
            # If the record exists but does not have admin role, promote it and commit
            if admin_exists.role != 'admin':
                old_role = admin_exists.role
                admin_exists.role = 'admin'
                db.session.commit()
                print(f"Startup DEBUG: Existing user '{admin_exists.registration_no}' promoted from role '{old_role}' to 'admin'.")
            else:
                print(f"Startup DEBUG: Admin user '{admin_exists.registration_no}' already exists. Role: '{admin_exists.role}'")

        hod_exists = User.query.filter_by(registration_no='HOD001').first()
        if not hod_exists:
            hod_user = User(registration_no='HOD001', full_name='HOD User', email='hod@example.com', role='hod')
            hod_user.set_password('hodpassword') # Use set_password method
            db.session.add(hod_user)
            db.session.commit()
            print("Startup DEBUG: Default HOD user created: HOD001 / hodpassword")
        else:
            # If the record exists but does not have hod role, promote it and commit
            if hod_exists.role != 'hod':
                old_role = hod_exists.role
                hod_exists.role = 'hod'
                db.session.commit()
                print(f"Startup DEBUG: Existing user '{hod_exists.registration_no}' promoted from role '{old_role}' to 'hod'.")
            else:
                print(f"Startup DEBUG: HOD user '{hod_exists.registration_no}' already exists. Role: '{hod_exists.role}'")

# Run DB initialization on import (for Gunicorn/Render)
init_db()

@app.route("/notifications")
@login_required
def notifications():
    """View all notifications for the current user."""
    notifs = Notification.query.filter_by(user_id=current_user.id).order_by(Notification.timestamp.desc()).all()
    # Mark all as read when viewing this page
    Notification.query.filter_by(user_id=current_user.id, is_read=False).update({Notification.is_read: True})
    db.session.commit()
    return render_template('notifications.html', title=_('Notifications'), notifications=notifs)

@app.route("/api/notifications/unread_count")
@login_required
def unread_notification_count():
    """Returns the number of unread notifications for the current user."""
    count = Notification.query.filter_by(user_id=current_user.id, is_read=False).count()
    return jsonify({"count": count})

@app.route("/admin/analytics/data")
@login_required
def admin_analytics_data():
    """Returns analytics data for the dashboard charts as JSON."""
    if not (current_user.is_admin() or current_user.is_hod()):
        abort(403)
    
    analytics_objs = get_analytics_summary(30)
    
    data = []
    for a in analytics_objs:
        data.append({
            "date": a.date.isoformat(),
            "lost": a.items_lost_count,
            "found": a.items_found_count,
            "returned": a.items_returned_count,
            "new_users": a.new_users_count,
            "active_users": a.active_users_count,
            "logins": a.total_logins,
            "success_rate": a.success_rate,
            "peak_hours": json.loads(a.peak_hours) if a.peak_hours else {}
        })
    
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)