Face Swap API
A FastAPI-based backend for face-swapping using a Gradio client. Deployed on Railway.app.
Setup

Clone the repository:
git clone <your-repo-url>
cd face-swap-api


Install dependencies:
pip install -r requirements.txt


Run locally:
uvicorn app:app --reload


Access the API at http://localhost:8000/docs.


Deployment
Deploy to Railway.app by linking the GitHub repository and setting environment variables:

UPLOAD_FOLDER: Directory for uploaded images (default: static/uploads)
OUTPUT_FOLDER: Directory for processed images (default: static/output)
MAX_FILE_SIZE: Max file size in bytes (default: 5242880 for 5MB)

API Endpoints

GET /: Renders the index page with a form for testing.
POST /swap: Uploads source and destination images for face-swapping.
Request: multipart/form-data with source_image and dest_image.
Response: JSON with success, data.result_image, and error.



License
MIT
