# Lost and Found Web Portal for Students

A professional web application designed to help students report and find lost or found items within a college/university campus. The portal allows students to register, log in, report items with detailed descriptions and images, and browse existing lost/found listings. Admin/HOD users have elevated privileges to manage (soft-delete) reported items, maintaining a clear history.

## Features

* **Student Registration & Login:** Secure authentication using registration number and password.
* **Report Lost Items:** Students can report items they've lost, including name, description, date, and location.
* **Report Found Items:** Students can report items they've found, including name, description, date, location, and **an image upload**.
* **View Lost & Found Items:** Browse a list of all active items.
* **Item Details:** Click on an item to view its full details, including uploaded image (for found items).
* **Role-Based Access Control (RBAC):**
    * **Students:** Can report, view, and manage their own active items.
    * **Admin/HOD:** Can view all items (active and history), and **soft-delete** items to move them to history.
* **Item History:** A dedicated section (accessible to Admin/HOD) to view all historical (soft-deleted) items with audit trail (who deleted, when).
* **Clear Audit Trail:** Each item entry includes details on who reported it, when, and if/when/who soft-deleted it.
* **User-Friendly Interface:** Built with HTML and CSS (using Bootstrap for responsive design).

## Technologies Used

* **Backend:** Python
    * **Framework:** Flask
    * **Database:** SQLAlchemy (ORM) with SQLite (development)
    * **Authentication:** Flask-Login
    * **Forms:** Flask-WTF
    * **Image Handling:** Pillow (PIL Fork) for processing, `werkzeug.utils` for secure filenames.
    * **Password Hashing:** `werkzeug.security`
* **Frontend:**
    * HTML5
    * CSS3 (with Bootstrap 4)
    * JavaScript (for image preview)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url_here>
    cd lost_found_app
    ```

2.  **Create a Python Virtual Environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies:**
    ```bash
    # Install core Python dependencies listed in the project file
    pip install -r requirement.txt

    # Additional CLIP dependencies (optional but recommended for image-text verification):
    python -m pip install transformers pillow

    # Install PyTorch (required to run CLIP efficiently). Choose the right command for your platform from the official guide:
    # https://pytorch.org/get-started/locally/
    # Example (CPU-only wheel on Windows or Linux):
    # python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

    # Optional (recommended) - YOLOv8 object detection for stronger detection/conflict checks:
    pip install ultralytics opencv-python
    # YOLOv8 will download the model weights (yolov8n.pt) automatically on first use.
    ```

5.  **Set Environment Variables (Optional but Recommended for Production):**
    For development, you can skip this, but in production, set these for security.
    * `SECRET_KEY`: A long, random string.
    * `DATABASE_URL`: Your database connection string (e.g., `postgresql://user:password@host:port/dbname`).
    ```bash
    # Example for Linux/macOS
    export SECRET_KEY='your_super_long_random_secret_key'
    export DATABASE_URL='sqlite:///site.db' # Or your PostgreSQL/MySQL URL
    ```
    (For Windows, use `set` instead of `export`)

6.  **Run the Application:**
    ```bash
    python app.py
    ```

    The application will be accessible at `http://127.0.0.1:5000/`.

7.  **Initial Admin/HOD Users:**
    Upon first run, the application will check and create default Admin and HOD users if they don't exist:
    * **Admin:** Registration No. `ADMIN001`, Password: `adminpassword`
    * **HOD:** Registration No. `HOD001`, Password: `hodpassword`
    **IMPORTANT: Change these default passwords immediately after initial setup in a real deployment!**

## Project Structure