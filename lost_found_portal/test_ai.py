
import os
import sys

# Add project root to sys.path to import app
sys.path.append(os.getcwd())

from app import app, _load_yolo, _load_clip_model, detect_with_yolo, clip_verify

def test_ai():
    # Use one of the existing images in uploads
    upload_dir = os.path.join('static', 'uploads')
    files = [f for f in os.listdir(upload_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not files:
        print("No images found in uploads to test.")
        return
    
    test_image = os.path.join(upload_dir, files[0])
    print(f"Testing with image: {test_image}")
    
    print("\n1. Testing YOLO...")
    try:
        yolo_results = detect_with_yolo(test_image)
        print(f"YOLO Results: {yolo_results}")
    except Exception as e:
        print(f"YOLO Failed: {e}")
        
    print("\n2. Testing CLIP...")
    try:
        with app.app_context():
            # Test with a common name
            ok, info, best, score = clip_verify("phone", test_image)
            print(f"CLIP Results: ok={ok}, info={info}, best={best}, score={score}")
    except Exception as e:
        print(f"CLIP Failed: {e}")

if __name__ == "__main__":
    test_ai()
