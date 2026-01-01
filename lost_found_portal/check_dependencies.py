"""
Dependency Checker for Lost & Found Portal
Checks if all required AI/ML dependencies are installed and working.
"""

import sys

def check_dependency(name, import_statement, test_func=None):
    """Check if a dependency is installed and working."""
    try:
        exec(import_statement)
        if test_func:
            test_func()
        print(f"✅ {name}: OK")
        return True
    except ImportError as e:
        print(f"❌ {name}: NOT INSTALLED - {e}")
        return False
    except Exception as e:
        print(f"⚠️  {name}: INSTALLED but ERROR - {e}")
        return False

def main():
    print("=" * 60)
    print("Lost & Found Portal - Dependency Check")
    print("=" * 60)
    print()
    
    dependencies = []
    
    # Core dependencies
    print("Core Dependencies:")
    print("-" * 60)
    dependencies.append(check_dependency("Flask", "import flask"))
    dependencies.append(check_dependency("SQLAlchemy", "from flask_sqlalchemy import SQLAlchemy"))
    dependencies.append(check_dependency("Flask-Login", "from flask_login import LoginManager"))
    dependencies.append(check_dependency("WTForms", "from wtforms import StringField"))
    dependencies.append(check_dependency("Passlib", "from passlib.context import CryptContext"))
    print()
    
    # Image processing
    print("Image Processing:")
    print("-" * 60)
    dependencies.append(check_dependency("Pillow (PIL)", "from PIL import Image"))
    print()
    
    # AI/ML dependencies
    print("AI/ML Dependencies:")
    print("-" * 60)
    
    # EasyOCR
    def test_easyocr():
        import easyocr
        # Don't actually load the reader, just check import
        pass
    dependencies.append(check_dependency("EasyOCR", "import easyocr", test_easyocr))
    
    # Ultralytics YOLO
    def test_yolo():
        from ultralytics import YOLO
        # Don't load model, just check import
        pass
    dependencies.append(check_dependency("Ultralytics (YOLO)", "from ultralytics import YOLO", test_yolo))
    
    # PyTorch
    def test_torch():
        import torch
        print(f"   PyTorch version: {torch.__version__}")
    dependencies.append(check_dependency("PyTorch", "import torch", test_torch))
    
    # Transformers (for CLIP)
    def test_transformers():
        from transformers import CLIPProcessor, CLIPModel
        pass
    dependencies.append(check_dependency("Transformers (CLIP)", "from transformers import CLIPProcessor, CLIPModel", test_transformers))
    
    print()
    print("=" * 60)
    print(f"Summary: {sum(dependencies)}/{len(dependencies)} dependencies OK")
    print("=" * 60)
    
    if not all(dependencies):
        print()
        print("⚠️  Some dependencies are missing or not working properly.")
        print("   The application may encounter errors when using AI features.")
        print()
        print("To install missing dependencies, run:")
        print("   pip install -r requirements.txt")
        return 1
    else:
        print()
        print("✅ All dependencies are installed and working!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
