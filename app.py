import os
import uuid
import hashlib
import tempfile
import time
import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance
from cachetools import TTLCache
from gradio_client import Client, handle_file
from retry import retry
from starlette.responses import Response

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS Middleware for API endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "static/uploads")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "static/output")
STATIC_DIR = "static"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 5 * 1024 * 1024))  # 5MB default

# Create static directories if they don't exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Custom StaticFiles class to add CORS headers
class CORSStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        # Add CORS headers to static file responses
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response

# Mount static files with CORS support
app.mount("/static", CORSStaticFiles(directory=STATIC_DIR), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Cache setup (TTL: 1 hour)
cache = TTLCache(maxsize=100, ttl=3600)

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file(file_path: str) -> bool:
    """Validate if the file exists and is an image."""
    if not os.path.exists(file_path):
        return False
    if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return False
    return True

def get_file_hash(file_content: bytes) -> str:
    """Generate a hash for a file to use as cache key."""
    return hashlib.sha256(file_content).hexdigest()

def compress_image(content: bytes, max_size: int = 1024) -> bytes:
    """Compress image to reduce size while maintaining quality."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        img = Image.open(temp_file_path)
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as compressed_file:
            img.save(compressed_file.name, "PNG", optimize=True, quality=85)
            with open(compressed_file.name, "rb") as f:
                compressed_content = f.read()
        os.unlink(temp_file_path)
        os.unlink(compressed_file.name)
        return compressed_content
    except Exception as e:
        logger.error(f"Image compression failed: {str(e)}")
        return content

def enhance_image(image_path: str) -> None:
    """Enhance image quality using sharpening."""
    try:
        img = Image.open(image_path)
        enhancer = ImageEnhance.Sharpness(img)
        img_enhanced = enhancer.enhance(2.0)
        img_enhanced.save(image_path, "PNG")
    except Exception as e:
        logger.error(f"Image enhancement failed: {str(e)}")

def save_output_image(result_path: str, output_dir: str, output_name: str) -> str:
    """Save the result image to a specified directory and enhance it."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)
        img = Image.open(result_path)
        img = img.convert("RGB")
        img.save(output_path, "PNG")
        enhance_image(output_path)
        return output_path
    except Exception as e:
        logger.error(f"Failed to save output image: {str(e)}")
        return ""

def cleanup_output_folder(max_age_seconds: int = 3600):
    """Remove files older than max_age_seconds from OUTPUT_FOLDER."""
    try:
        now = time.time()
        for filename in os.listdir(OUTPUT_FOLDER):
            file_path = os.path.join(OUTPUT_FOLDER, filename)
            if os.path.isfile(file_path) and (now - os.path.getmtime(file_path)) > max_age_seconds:
                os.remove(file_path)
    except Exception as e:
        logger.error(f"Output folder cleanup failed: {str(e)}")

@retry(tries=3, delay=2, backoff=2)
async def face_swap(source_image: str, dest_image: str, source_face_idx: int = 1, dest_face_idx: int = 1) -> str:
    """Perform face swap using Gradio Client with retry logic."""
    try:
        if not all([validate_file(source_image), validate_file(dest_image)]):
            return "Invalid input files"

        client = Client("Dentro/face-swap")
        result = client.predict(
            sourceImage=handle_file(source_image),
            sourceFaceIndex=source_face_idx,
            destinationImage=handle_file(dest_image),
            destinationFaceIndex=dest_face_idx,
            api_name="/predict"
        )

        if result and os.path.exists(result):
            unique_filename = f"face_swap_{uuid.uuid4().hex}.png"
            final_path = save_output_image(result, OUTPUT_FOLDER, unique_filename)
            if final_path:
                return final_path
            return "Failed to save output"
        return "Face swap failed"
    except Exception as e:
        logger.error(f"Face swap failed: {str(e)}")
        return f"Error: {str(e)}"

@app.get("/", description="Render the index page for the face-swap UI")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result_image": None})

@app.post("/swap", description="Upload source and destination images for face-swapping")
async def swap_faces(
    source_image: UploadFile = File(...),
    dest_image: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Swap faces between two uploaded images."""
    logger.info("Received request to swap faces")
    
    # Validate file uploads
    if not source_image.filename or not dest_image.filename:
        logger.error("No file selected")
        return JSONResponse(
            status_code=400,
            content={"success": False, "data": None, "error": "No file selected"}
        )

    if not (allowed_file(source_image.filename) and allowed_file(dest_image.filename)):
        logger.error("Invalid file format")
        return JSONResponse(
            status_code=400,
            content={"success": False, "data": None, "error": "Invalid file format. Only PNG, JPG, JPEG allowed"}
        )

    # Validate file size
    source_content = await source_image.read()
    dest_content = await dest_image.read()
    source_size = len(source_content)
    dest_size = len(dest_content)
    if source_size > MAX_FILE_SIZE or dest_size > MAX_FILE_SIZE:
        logger.error(f"File size exceeds limit: {MAX_FILE_SIZE / (1024 * 1024)}MB")
        return JSONResponse(
            status_code=400,
            content={"success": False, "data": None, "error": f"File size exceeds {MAX_FILE_SIZE / (1024 * 1024)}MB"}
        )

    # Compress file contents
    logger.info("Compressing images")
    source_content = compress_image(source_content)
    dest_content = compress_image(dest_content)

    # Generate cache key
    cache_key = f"{get_file_hash(source_content)}:{get_file_hash(dest_content)}"

    # Check cache
    if cache_key in cache:
        result_url = f"/{cache[cache_key]}"
        logger.info(f"Cache hit: {result_url}")
        background_tasks.add_task(cleanup_output_folder)
        return {"success": True, "data": {"result_image": result_url}, "error": None}

    # Save files temporarily
    with tempfile.TemporaryDirectory() as temp_dir:
        source_filename = f"source_{uuid.uuid4().hex}.{source_image.filename.rsplit('.', 1)[1]}"
        dest_filename = f"dest_{uuid.uuid4().hex}.{dest_image.filename.rsplit('.', 1)[1]}"
        source_path = os.path.join(temp_dir, source_filename)
        dest_path = os.path.join(temp_dir, dest_filename)

        # Write compressed files
        logger.info("Writing temporary files")
        with open(source_path, "wb") as f:
            f.write(source_content)
        with open(dest_path, "wb") as f:
            f.write(dest_content)

        # Perform face swap
        logger.info("Performing face swap")
        result = await face_swap(source_path, dest_path)
        if result.startswith("Error") or result in ["Invalid input files", "Failed to save output"]:
            logger.error(f"Face swap error: {result}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "data": None, "error": result}
            )

        # Cache result
        cache[cache_key] = result
        logger.info(f"Cached result: {result}")

        # Schedule cleanup
        background_tasks.add_task(cleanup_output_folder)

        # Return result
        result_url = f"/{result}"
        logger.info(f"Returning result: {result_url}")
        return {"success": True, "data": {"result_image": result_url}, "error": None}
