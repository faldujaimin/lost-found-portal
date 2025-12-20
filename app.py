from flask import Flask, render_template, redirect, url_for, flash, request, jsonify
from flask import abort
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import check_password_hash
from passlib.context import CryptContext
from datetime import datetime
import os
import sqlite3
import shutil
from datetime import datetime as _dt
from sqlalchemy import event
from sqlalchemy.engine import Engine
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TextAreaField, DateField, FileField, BooleanField
from wtforms.validators import DataRequired, Length, EqualTo, ValidationError, Regexp, Optional
from wtforms.widgets import DateInput
from flask_wtf.file import FileAllowed

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
    'phone','mobile','iphone','android','samsung','xiaomi','pixel','phone','handphone',
    'book','bag','backpack','purse','wallet','keys','key','id','id card','card',
    'laptop','macbook','notebook','watch','ring','jewel','jewelry','glasses','specs','spectacles',
    'umbrella','pen','pens','pen drive','pendrive','usb','flash drive','calculator','calc','charger',
    'cable','wire','bottle','water bottle','mouse','file','files','document','documents','chain',
    'necklace','handsfree','hands-free','headphone','headphones','earphones','earbuds','head set','head-set'
]
 
app = Flask(__name__)
# IMPORTANT: Change this to a strong, unique, and secret key in production!
app.config['SECRET_KEY'] = 'your_super_secret_key_change_this_in_production_really_strong'
# Use an absolute path to the SQLite file inside instance/ so it isn't lost if
# the server process is restarted or run from a different working directory.
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + DB_PATH.replace('\\', '/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # Link upload folder to Flask config
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Limit uploads to 16 Megabytes
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS # Store allowed extensions in config
# Toggle CLIP-based image-text verification (set to False to disable)
app.config['USE_CLIP_VERIFICATION'] = True
# Default CLIP matching threshold (soft, tuneable per-site)
app.config['CLIP_THRESHOLD'] = 0.22
# Thresholds for detecting strong conflicting signals
app.config['DETECTION_CONFLICT_THRESHOLD'] = 0.7
app.config['CLIP_CONFLICT_THRESHOLD'] = 0.6
# Note: history page is no longer protected by a secondary password. Access is
# controlled solely by user role (admin or hod).

# Initialize SQLAlchemy with the Flask app instance
db = SQLAlchemy(app)
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
# ...existing code...
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
        # This prevents 100% confidence when there is only 1 candidate
        common_objects = [
            'cell phone', 'smartphone', 'laptop', 'computer', 
            'backpack', 'bag', 'handbag', 'wallet', 'purse',
            'keys', 'keychain', 'watch', 'wrist watch',
            'headphones', 'earbuds', 'water bottle', 'umbrella',
            'id card', 'credit card', 'glasses', 'sunglasses',
            'book', 'notebook', 'shoe', 'clothing'
        ]
        
        for obj in common_objects:
            if obj not in candidates and obj not in name.lower():
                candidates.append(obj)

        inputs = processor(text=candidates, images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
        
        # Get score for the user's item name (or best matching part of it)
        # We look for the index of the candidate that matches the item_name (or is a part of it)
        user_candidate_indices = [i for i, c in enumerate(candidates) if c in name or name in c]
        
        if not user_candidate_indices:
             # Should happen rarely as we added name to candidates
             user_candidate_indices = [0]
             
        # The score for the "claimed item" is the max probability among its variations
        claimed_score = max([float(probs[i]) for i in user_candidate_indices])
        
        # find best match overall
        best_idx = int(probs.argmax())
        best_score = float(probs[best_idx])
        best_label = candidates[best_idx]
        
        print(f"DEBUG: CLIP: best match '{best_label}' score={best_score:.3f}, claimed '{item_name}' score={claimed_score:.3f}")
        
        # We return the score of the CLAIMED item, but validity is based on whether it's close to the best match
        if claimed_score >= threshold and (claimed_score >= best_score * 0.8):
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
        
        # 1. Run YOLO
        try:
            yolo_dets = detect_with_yolo(temp_path)
            results['detected_objects'] = yolo_dets
            if yolo_dets:
                results['best_guess'] = yolo_dets[0]['label']
                results['confidence'] = yolo_dets[0]['score']
        except Exception as e:
            print(f"YOLO failed: {e}")
            
        # 2. Run CLIP with common objects if YOLO failed or for better context
        # We compare against the common list defined above
        try:
             # Re-use logic or call a simplified clip helper
             # For now, just rely on YOLO for explicit detection display, or add basic CLIP classification if YOLO is empty
             if not results['detected_objects']:
                 # Simplified CLIP classification
                 model_proc = _load_clip_model()
                 if model_proc:
                     model, processor = model_proc
                     from PIL import Image
                     import torch
                     img = Image.open(temp_path).convert('RGB')
                     
                     common_objects = [
                        'cell phone', 'laptop', 'bag', 'wallet', 'watch', 
                        'keys', 'headphones', 'bottle', 'id card', 'book'
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
            
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# Lazy YOLOv8 loader and detector (Ultralytics)
_yolo_model = None

def _load_yolo():
    global _yolo_model
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


def detect_with_yolo(image_path, conf=0.25):
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
    'watch': [],
    'ring': [],
    'glasses': [],
    'specs': [],
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
{'laptop','macbook','notebook','wallet','id','id card','card','backpack','bag','purse'}



def verify_protected_image(item_name, file_path, score_threshold=0.3, min_size=120):
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
        yolo_dets = detect_with_yolo(file_path)
        if yolo_dets:
            detection_attempted = True
            print(f"DEBUG: YOLO labels for '{item_name}': {yolo_dets}")
            # If YOLO finds a matching expected label, accept
            for exp in expected:
                for d in yolo_dets:
                    if exp in d['label'] or d['label'] in exp:
                        return (True, f"YOLO detected: {d['label']} ({d['score']:.2f})")
            # If YOLO top detection is a strong conflicting object, reject
            top = yolo_dets[0]
            conflict_thresh = app.config.get('DETECTION_CONFLICT_THRESHOLD', 0.7)
            if top['score'] >= conflict_thresh:
                reason = f"Uploaded image appears to show '{top['label']}' (confidence {top['score']:.2f}), not '{item_name}'. Please upload a photo of the claimed item or correct the item name."
                print(f"DEBUG: YOLO detection conflict: {reason}")
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


def cross_verify_item(item_name, file_path, extracted_text=None):
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
        yolo_detections = detect_with_yolo(file_path)
        if yolo_detections:
            # Check if any YOLO detection matches item name
            best_yolo_score = 0.0
            best_yolo_label = None
            
            for detection in yolo_detections:
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
            details['yolo_detections'] = [{'label': d['label'], 'score': d['score']} for d in yolo_detections[:3]]
            
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
            'bag': ['bag', 'backpack', 'handbag', 'purse', 'suitcase', 'luggage'],
            'wallet': ['wallet', 'purse'],
            'watch': ['watch', 'clock', 'wristwatch'],
            'keys': ['keys', 'key'],
            'card': ['card', 'id', 'license'],
            'book': ['book', 'notebook', 'textbook']
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
            for obj in detected_objects:
                for category, keywords in object_categories.items():
                    if category != claimed_category and any(kw in obj for kw in keywords):
                        detected_category = category
                        conflict_penalty = 0.7  # Heavy penalty for mismatched category
                        details['conflict'] = f'Claimed {claimed_category} but detected {category}'
                        print(f"DEBUG: CONFLICT DETECTED - Claimed: {claimed_category}, Detected: {category}")
                        break
                if detected_category:
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


# --- Models (Object-Relational Mapping) ---
# IMPORTANT: Item model is defined BEFORE User model to resolve
# "forward reference" issues for foreign_keys in User's relationships.
class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(10), default='Lost', nullable=False) # 'Lost', 'Found', 'Reclaimed'
    reported_at = db.Column(db.DateTime, default=datetime.utcnow) # Timestamp when reported
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
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    details = db.Column(db.Text, nullable=True)

    user = db.relationship('User', backref='logs', foreign_keys=[user_id])
    item = db.relationship('Item', backref='logs', foreign_keys=[item_id])
    

    def __repr__(self):
        # Helpful representation for debugging in the shell or logs
        return f"ReportLog(id={self.id}, reg_no={self.registration_no}, action={self.action}, item_id={self.item_id}, ts={self.timestamp})"
def india_now():
    return datetime.now(TZ_INDIA) if TZ_INDIA else datetime.utcnow()

class AuthLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    registration_no = db.Column(db.String(20), nullable=True)
    action = db.Column(db.String(20), nullable=False)  # 'login' or 'logout'
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: india_now())
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
                    timestamp= india_now(),
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
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref='messages', foreign_keys=[user_id])
    item = db.relationship('Item', backref='messages', foreign_keys=[item_id])

    def __repr__(self):
        return f"Message(id={self.id}, item_id={self.item_id}, user_id={self.user_id}, ts={self.timestamp})"

# --- WTForms (Web Forms) ---
# Forms are defined AFTER models, as they might reference models (e.g., for validation)
class RegistrationForm(FlaskForm):
    registration_no = StringField('Registration No.', validators=[
        DataRequired(), Length(min=5, max=20),
        Regexp('^[A-Za-z0-9]+$', message="Registration number must contain only letters and digits.")
    ])
    full_name = StringField('Full Name', validators=[DataRequired(), Length(min=2, max=100)])
    email = StringField('Email (Optional)', validators=[Optional(), Length(max=120)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    # Custom validator to check if registration number already exists
    def validate_registration_no(self, registration_no):
        user = User.query.filter_by(registration_no=registration_no.data).first()
        if user:
            raise ValidationError('That registration number is already taken. Please choose a different one.')

    # Custom validator to check if email already exists
    def validate_email(self, email):
        if email.data and email.data.strip(): # Only validate if email is provided AND not empty/whitespace
            user = User.query.filter_by(email=email.data.strip()).first()
            if user:
                raise ValidationError('That email address is already registered. Please use a different one or log in.')


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
        DataRequired(),
        FileAllowed(ALLOWED_EXTENSIONS, 'Images only! (png, jpg, jpeg, gif)')
    ])
    submit = SubmitField('Report Found Item')
    # Optional confirmation for phone images; shown client-side when item is a phone
    phone_confirm = BooleanField('I confirm this image shows the phone')


class ProfilePhotoForm(FlaskForm):
    photo = FileField('Profile Photo', validators=[
        FileAllowed(ALLOWED_EXTENSIONS, 'Images only! (png, jpg, jpeg, gif)')
    ])
    submit = SubmitField('Upload')


class MessageForm(FlaskForm):
    content = TextAreaField('Message', validators=[DataRequired(), Length(min=1, max=1000)])
    submit = SubmitField('Send')


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
@app.route("/")
@app.route("/home")
def home():
    return render_template('index.html', title='Home')

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        flash('You are already logged in.', 'info')
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        email_data = form.email.data.strip() if form.email.data and form.email.data.strip() else None
        
        user = User(registration_no=form.registration_no.data,
                    full_name=form.full_name.data,
                    email=email_data,
                    role='student') # Explicitly set role to student for registrations
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        print(f"DEBUG: New student registered: Reg No: {user.registration_no}, Email: {user.email}")
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        flash('You are already logged in.', 'info')
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
                flash('Login successful!', 'success')
                print(f"DEBUG: Login successful for user '{user.registration_no}' ({user.role}). Redirecting to {next_page or url_for('items')}")
                return redirect(next_page) if next_page else redirect(url_for('items'))
            else:
                print(f"DEBUG: Password mismatch for user '{user.registration_no}'.")
                flash('Login Unsuccessful. Please check registration number and password', 'danger')
        else:
            print(f"DEBUG: No user found with Registration No: '{attempted_reg_no}'.")
            flash('Login Unsuccessful. Please check registration number and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route("/logout")
@login_required
def logout():
    print(f"DEBUG: User '{current_user.registration_no}' logging out.")
    logout_user()
    flash('You have been logged out.', 'info')
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
        flash('Your lost item report has been submitted!', 'success')
        print(f"DEBUG: Lost item reported by {current_user.registration_no}: {item.item_name}")
        return redirect(url_for('items'))
    return render_template('report_item_modern.html', title='Report Lost Item', form=form, item_type='Lost', protected_keywords=PROTECTED_KEYWORDS)

@app.route("/report_found", methods=['GET', 'POST'])
@login_required
def report_found():
    form = ReportFoundItemForm()
    if form.validate_on_submit():
        if 'image' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        # Server-side check: if item appears to be a protected/valuable type, ensure user ticked confirmation
        def looks_like_protected(name):
            if not name: return False
            ln = name.lower()
            return any(k in ln for k in PROTECTED_KEYWORDS)

        if looks_like_protected(form.item_name.data) and not form.phone_confirm.data:
            flash('This item appears to be a protected or valuable item. Please confirm you have the right to upload this photo, check the confirmation box, and upload a clear close-up photo (portrait or closer framing).', 'warning')
            return redirect(request.url)

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
                    flash(f"Error creating upload directory: {e}. Check server permissions.", 'danger')
                    return redirect(request.url)
            else:
                print(f"DEBUG: Directory '{upload_path}' already exists.")

            if not os.access(upload_path, os.W_OK):
                print(f"ERROR: Directory '{upload_path}' is not writable by the current user.")
                flash(f"Upload directory is not writable. Check server permissions for '{upload_path}'.", 'danger')
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
                    # If protected item (e.g., phone/laptop) and image is unusually wide, warn the user
                    if looks_like_protected(form.item_name.data) and (w > h * 1.5):
                        # For strict items, reject wide/odd images rather than just warn
                        if any(k in form.item_name.data.lower() for k in PROTECTED_STRICT):
                            # remove saved file
                            try:
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                                    print(f"DEBUG: Removed image due to unacceptable aspect ratio: {file_path}")
                            except Exception as e:
                                print(f"DEBUG: Failed to remove file after aspect rejection: {e}")
                            flash('The uploaded image is not an acceptable close-up of this protected item. Please upload a clear, portrait/close-up image of the claimed item and try again.', 'warning')
                            return redirect(request.url)
                        else:
                            flash('Uploaded image seems unusually wide for this item. Please upload a clear photo (portrait or closer framing).', 'warning')

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

                # If the item name looks like a protected/valuable type, run server-side verification
                if looks_like_protected(form.item_name.data):
                    ok, info = verify_protected_image(form.item_name.data, file_path)
                    if not ok:
                        # Remove the saved file to avoid storing unverified images
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                print(f"DEBUG: Removed unverified uploaded file: {file_path}")
                        except Exception as e:
                            print(f"DEBUG: Failed to remove unverified file: {e}")

                        # Show a helpful message and include the verification reason to guide the user
                        base = 'The uploaded image does not appear to show the claimed item. Please upload a clear photo that shows the item (close-up or different angle) and try again.'
                        reason = info if isinstance(info, str) else str(info)
                        msg = f"{base} Reason: {reason}"
                        flash(msg, 'warning')
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
                        extracted_text=ocr_text
                    )
                    verification_score = confidence
                    verification_details_dict = details
                    print(f"DEBUG: Cross-verification - Confidence: {confidence:.2f}, Verified: {verified}")
                    
                    # Check for conflicts
                    has_conflict = details.get('conflict_penalty', 0) > 0
                    conflict_msg = details.get('conflict', '')
                    
                    # Reject extremely low confidence (likely mislabeled)
                    if confidence < 0.2 or has_conflict:
                        # Remove the saved file
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                print(f"DEBUG: Removed file due to low confidence/conflict: {file_path}")
                        except Exception as e:
                            print(f"DEBUG: Failed to remove file: {e}")
                        
                        # Show detailed error message
                        if conflict_msg:
                            error_msg = f'Item name does not match the image. {conflict_msg}. Please ensure the item name accurately describes what is in the photo.'
                        else:
                            error_msg = f'Very low AI confidence ({int(confidence*100)}%). The image does not appear to match the item name. Please verify you uploaded the correct photo and item name.'
                        
                        flash(error_msg, 'danger')
                        return redirect(request.url)
                    
                    # Show verification feedback to user for successful submissions
                    if confidence >= 0.7:
                        flash(f'✅ High confidence match ({int(confidence*100)}%) - Item verified!', 'success')
                    elif confidence >= 0.4:
                        flash(f'⚠️ Medium confidence ({int(confidence*100)}%) - Please verify the details are correct.', 'warning')
                    elif confidence >= 0.2:
                        flash(f'⚠️ Low confidence match ({int(confidence*100)}%) - Please double-check item name and image match.', 'warning')
                        
                except Exception as e:
                    print(f"DEBUG: Cross-verification error: {e}")
            except Exception as e:
                print(f"ERROR: Failed to save file '{filename}' to '{file_path}': {e}")
                flash(f"Error saving image: {e}. Check server disk space or permissions.", 'danger')
                return redirect(request.url)

            # Store verification details as JSON
            import json
            verification_json = json.dumps(verification_details_dict) if verification_details_dict else None

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
            db.session.commit()
            # Log the found report
            log = ReportLog(user_id=current_user.id, registration_no=current_user.registration_no,
                            action='reported_found', item_id=item.id,
                            details=f"Found report: {item.item_name} (Image: {filename})")
            db.session.add(log)
            db.session.commit()
            # After saving the found report, check for matching active lost reports
            matching_lost = Item.query.filter(
                Item.status == 'Lost',
                Item.is_active == True,
                Item.item_name.ilike(f"%{item.item_name}%")
            ).all()
            points_awarded = 0
            if matching_lost:
                # Award 1 point per matched lost report (previously 10 points each)
                points_awarded = len(matching_lost)
                current_user.points = (current_user.points or 0) + points_awarded
                current_user.update_level()
                db.session.commit()
                # Log the point award
                plog = ReportLog(user_id=current_user.id, registration_no=current_user.registration_no,
                                action='points_awarded', item_id=item.id,
                                details=f"Awarded {points_awarded} points for matching {len(matching_lost)} lost report(s)")
                db.session.add(plog)
                db.session.commit()

            flash('Your found item report has been submitted!' + (f" You earned {points_awarded} points!" if points_awarded else ''), 'success')
            print(f"DEBUG: Found item reported by {current_user.registration_no}: {item.item_name} (Image: {item.image_filename}). Points awarded: {points_awarded}")

            return redirect(url_for('items'))
        else:
            flash(f"Invalid file type. Allowed: {', '.join(app.config['ALLOWED_EXTENSIONS'])}", 'danger')
    return render_template('report_item_modern.html', title='Report Found Item', form=form, item_type='Found', protected_keywords=PROTECTED_KEYWORDS)

@app.route("/items")
@login_required
def items():
    # Support optional search via ?q=term
    q = request.args.get('q', '').strip()
    if q:
        # Filter by name, description or location (case-insensitive)
        from sqlalchemy import or_
        active_items = Item.query.filter(
            Item.is_active == True,
            or_(
                Item.item_name.ilike(f"%{q}%"),
                Item.description.ilike(f"%{q}%"),
                Item.location.ilike(f"%{q}%")
            )
        ).order_by(Item.reported_at.desc()).all()
    else:
        active_items = Item.query.filter_by(is_active=True).order_by(Item.reported_at.desc()).all()
    print(f"DEBUG: Displaying {len(active_items)} active items for {current_user.registration_no}. Query='{q}'")
    return render_template('item_list.html', title='Lost & Found Items', items=active_items)

@app.route("/item/<int:item_id>", methods=['GET', 'POST'])
@login_required
def item_detail(item_id):
    item = db.session.get(Item, item_id)
    if item is None:
        # Item missing  return 404 to the user
        print(f"DEBUG: Item with ID {item_id} not found for detail view.")
        return render_template('404.html', title='Item Not Found'), 404

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
        flash('Message posted.', 'success')
        return redirect(url_for('item_detail', item_id=item.id))

    # Load messages for this item (most recent last)
    messages = Message.query.filter_by(item_id=item.id).order_by(Message.timestamp.asc()).all()
    print(f"DEBUG: Displaying details for item ID {item_id}: {item.item_name} with {len(messages)} messages")
    return render_template('item_detail_modern.html', title='Item Details', item=item, form=form, messages=messages)


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
        flash('Item not found.', 'danger')
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
        flash('Item has been successfully moved to history.', 'success')
        print(f"DEBUG: Item '{item.item_name}' (ID: {item_id}) soft-deleted by {current_user.registration_no}.")
    else:
        flash('Item is already in history.', 'info')
        print(f"DEBUG: Item '{item.item_name}' (ID: {item_id}) already inactive, no action taken.")
    return redirect(url_for('items'))


@app.context_processor
def utility_processor():
    # Expose a helper to templates to check delete permission
    def can_delete_item(user):
        if not user:
            return False
        return user.is_admin() or user.is_hod()
    return dict(can_delete_item=can_delete_item)


@app.errorhandler(403)
def forbidden(error):
    # Provide a friendly forbidden page and log the event
    print(f"DEBUG: 403 Forbidden - {request.remote_addr} attempted an unauthorized action. Message: {error}")
    return render_template('403.html', title='Forbidden'), 403

@app.route("/history", methods=['GET', 'POST'])
@login_required
def item_history():
    # Only admin or hod can attempt to view history
    if not (current_user.is_admin() or current_user.is_hod()):
        flash('You do not have permission to view item history.', 'danger')
        print(f"DEBUG: User {current_user.registration_no} (Role: {current_user.role}) attempted unauthorized access to history.")
        return redirect(url_for('items'))

    # Only users with admin or hod roles can view history — render it directly.
    all_items = Item.query.order_by(Item.reported_at.desc()).all()
    print(f"DEBUG: Displaying {len(all_items)} items (including inactive) for {current_user.registration_no}.")
    return render_template('item_history.html', title='Item History', items=all_items)

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
            flash('No file selected.', 'warning')
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
                    flash(f'Unable to create upload directory: {e}', 'danger')
                    return redirect(url_for('edit_profile'))

            file_path = os.path.join(upload_path, save_name)
            try:
                file.save(file_path)
            except Exception as e:
                flash(f'Failed to save file: {e}', 'danger')
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
            flash('Profile photo updated.', 'success')
            return redirect(url_for('profile'))
        else:
            flash('Invalid file type.', 'danger')
            return redirect(url_for('edit_profile'))

    return render_template('edit_profile.html', title='Edit Profile', form=form, user=current_user)


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
    return render_template('levels.html', title='Your Level', user=user)


@app.route("/leaderboard")
@login_required
def leaderboard():
    # Show top users by points
    top_users = User.query.order_by(User.points.desc()).limit(50).all()
    return render_template('leaderboard.html', title='Leaderboard', users=top_users)


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

    return render_template('profile.html', title='Your Profile', user=user, reports=recent_reports,
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


if __name__ == '__main__':
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

        # Ensure any newly added columns exist in the SQLite 'item' table (useful when adding features)
        def ensure_item_columns():
            """Add missing columns to the 'item' table for older databases.
            This is a lightweight migration helper that runs only when needed.
            """
            conn = None
            try:
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                cur.execute("PRAGMA table_info('item')")
                existing = set(r[1] for r in cur.fetchall())
                # Desired columns mapped to their SQL definitions
                desired = {
                    'ocr_extracted_text': 'TEXT NULL',
                    'verification_score': 'REAL NULL',
                    'verification_details': 'TEXT NULL'
                }
                for col, definition in desired.items():
                    if col not in existing:
                        try:
                            sql = f"ALTER TABLE item ADD COLUMN {col} {definition};"
                            cur.execute(sql)
                            print(f"Startup DEBUG: Added missing column '{col}' to 'item' table.")
                        except Exception as e:
                            print(f"Startup ERROR: Failed to add column {col}: {e}")
                conn.commit()
            except Exception as e:
                print(f"Startup ERROR: ensure_item_columns failed: {e}")
            finally:
                if conn:
                    conn.close()

        ensure_item_columns()

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
    
    app.run(debug=True)