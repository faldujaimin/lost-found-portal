# Deploying to Render

This guide explains how to deploy the Lost & Found Portal to Render, specifically addressing the "500 Internal Server Error" caused by heavy AI model dependencies.

## 🚀 Quick Fix for 500 Error

The 500 error on Render is caused by the server running out of memory when trying to load large AI models (PyTorch, YOLO, etc.). We have added a "Lightweight Mode" that disables these features for deployment.

### Step 1: Update Environment Variables

In your Render Dashboard:
1. Go to your **Web Service**.
2. Click on **Environment**.
3. Add or Update the following variable:
   - Key: `ENABLE_AI_FEATURES`
   - Value: `false`

### Step 2: Use Lightweight Requirements

You need to tell Render to install the lightweight dependencies instead of the full AI suite.

**Option A: Update Build Command (Recommended)**
1. Go to **Settings** -> **Build & Deploy**.
2. Change the **Build Command** to:
   ```bash
   pip install -r requirements-render.txt
   ```
   *(Note: The default is usually `pip install -r requirements.txt`. changing it to `requirements-render.txt` installs only the essentials.)*

**Option B: Rename Files (If you can't change the build command)**
1. Rename `requirements.txt` to `requirements-dev.txt` (for local use).
2. Rename `requirements-render.txt` to `requirements.txt`.
3. Commit and push.

### Step 3: Verify Deployment

1. Trigger a manual deploy (Clear build cache & deploy) if needed.
2. Once deployed, visit your site.
3. Go to `/health` (e.g., `https://your-app.onrender.com/health`).
   - It should return: `{"ai_features": "disabled", "database": "ok", "status": "live"}`.

## 💡 What Changes in "Lightweight Mode"?

When `ENABLE_AI_FEATURES=false`:
- **Image Verification**: Basic checks only (file type, size). Advanced AI matching (YOLO, CLIP) is skipped.
- **Text Extraction**: OCR is disabled. Users must manually type item details.
- **Performance**: The app starts much faster and uses < 200MB RAM.

This allows your app to run smoothly on Render's free tier.
