import os
import uuid
import hashlib
import tempfile
import time
import logging
import requests
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks, HTTPException, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from cachetools import TTLCache
from gradio_client import Client, handle_file
from retry import retry
import asyncio
import shutil

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="High-Quality Face Swap API", version="2.3.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time"],
)

# Configuration
CONFIG = {
    "UPLOAD_FOLDER": os.getenv("UPLOAD_FOLDER", "static/uploads"),
    "OUTPUT_FOLDER": os.getenv("OUTPUT_FOLDER", "static/output"),
    "STATIC_DIR": "static",
    "ALLOWED_EXTENSIONS": {'png', 'jpg', 'jpeg', 'webp'},
    "MAX_FILE_SIZE": int(os.getenv("MAX_FILE_SIZE", 30 * 1024 * 1024)),  # 30MB
    "CACHE_TTL": int(os.getenv("CACHE_TTL", 7200)),  # 2 hours
    "MAX_CACHE_SIZE": int(os.getenv("MAX_CACHE_SIZE", 100)),
    "CLEANUP_INTERVAL": int(os.getenv("CLEANUP_INTERVAL", 86400)),  # 24 hours
    "MIN_IMAGE_SIZE": 10000,  # Minimum file size in bytes
    "GRADIO_TIMEOUT": int(os.getenv("GRADIO_TIMEOUT", 120)),  # Timeout for Gradio client
}

# Create directories and verify permissions
for folder in [CONFIG["STATIC_DIR"], CONFIG["UPLOAD_FOLDER"], CONFIG["OUTPUT_FOLDER"]]:
    os.makedirs(folder, exist_ok=True)
    try:
        test_file = os.path.join(folder, f"test_{uuid.uuid4().hex}.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.unlink(test_file)
        logger.info(f"Write permissions verified for {folder}")
    except Exception as e:
        logger.error(f"Failed to verify write permissions for {folder}: {str(e)}")
        raise Exception(f"Cannot write to {folder}: {str(e)}")

# Custom StaticFiles
class CORSStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        logger.debug(f"Serving static file: {path}")
        try:
            response = await super().get_response(path, scope)
            response.headers.update({
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Cache-Control": "public, max-age=3600",
                "Content-Disposition": "inline"
            })
            logger.debug(f"Static file served successfully: {path}")
            return response
        except Exception as e:
            logger.error(f"Failed to serve static file {path}: {str(e)}")
            raise HTTPException(404, detail=f"Static file {path} not found")

app.mount("/static", CORSStaticFiles(directory=CONFIG["STATIC_DIR"]), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Cache setup
cache = TTLCache(maxsize=CONFIG["MAX_CACHE_SIZE"], ttl=CONFIG["CACHE_TTL"])

# Progress tracking
progress_tracker: Dict[str, str] = {}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in CONFIG["ALLOWED_EXTENSIONS"]

def validate_image(file_path: str) -> bool:
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify image integrity
            if os.path.getsize(file_path) < CONFIG["MIN_IMAGE_SIZE"]:
                logger.error(f"Image too small or potentially corrupted: {file_path}")
                return False
        logger.debug(f"Image validated successfully: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Image validation failed for {file_path}: {str(e)}")
        return False

def validate_image_url(url: str) -> bool:
    try:
        response = requests.head(url, timeout=5)
        if response.status_code != 200:
            logger.error(f"Image URL not accessible: {url} (HTTP {response.status_code})")
            return False
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            logger.error(f"URL does not point to an image: {url} (Content-Type: {content_type})")
            return False
        content_length = int(response.headers.get("content-length", 0))
        if content_length < CONFIG["MIN_IMAGE_SIZE"]:
            logger.error(f"Image at URL {url} is too small ({content_length} bytes)")
            return False
        logger.debug(f"Image URL validated successfully: {url}")
        return True
    except requests.RequestException as e:
        logger.error(f"Failed to validate image URL {url}: {str(e)}")
        return False

def get_file_hash(file_content: bytes) -> str:
    return hashlib.sha256(file_content).hexdigest()

def get_image_extension(content: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        with Image.open(temp_file_path) as img:
            ext = img.format.lower()
        os.unlink(temp_file_path)
        return ext
    except Exception as e:
        logger.error(f"Failed to get image extension: {str(e)}")
        return "jpg"

async def high_quality_enhance(image_path: str) -> str:
    client = None
    try:
        logger.info(f"Starting enhancement for {image_path}")
        logger.debug(f"Initializing Tile-Upscaler client for {image_path}")
        client = Client("gokaygokay/Tile-Upscaler", httpx_kwargs={"timeout": CONFIG["GRADIO_TIMEOUT"]})
        
        logger.debug(f"Sending image {image_path} to Tile-Upscaler")
        result = client.predict(
            param_0=handle_file(image_path),
            param_1=512,
            param_2=20,
            param_3=0.4,
            param_4=0,
            param_5=3,
            api_name="/wrapper"
        )
        
        logger.debug(f"Gradio predict result: {result}")
        if not result or not isinstance(result, list) or len(result) < 2:
            logger.error(f"No valid enhanced image returned for {image_path}")
            raise ValueError("No valid enhanced image returned")
        
        enhanced_path = result[1]  # Use second result for the desired image
        logger.info(f"Gradio returned enhanced path: {enhanced_path}")
        
        if enhanced_path.startswith(('http://', 'https://')):
            gradio_url = enhanced_path
        else:
            gradio_url = f"https://gokaygokay-tile-upscaler.hf.space/file={enhanced_path}"
        
        if not validate_image_url(gradio_url):
            logger.error(f"Invalid or inaccessible Gradio image URL: {gradio_url}")
            raise ValueError(f"Invalid or inaccessible Gradio image URL: {gradio_url}")
        
        logger.info(f"Returning Gradio enhanced image URL: {gradio_url}")
        return gradio_url
        
    except Exception as e:
        logger.error(f"Image enhancement failed for {image_path}: {str(e)}")
        raise
    finally:
        if client:
            try:
                client.close()
                logger.debug(f"Tile-Upscaler client closed for {image_path}")
            except Exception as e:
                logger.warning(f"Failed to close Tile-Upscaler client: {str(e)}")

async def cleanup_output_folder():
    try:
        now = time.time()
        for filename in os.listdir(CONFIG["OUTPUT_FOLDER"]):
            file_path = os.path.join(CONFIG["OUTPUT_FOLDER"], filename)
            if os.path.isfile(file_path) and (now - os.path.getmtime(file_path)) > CONFIG["CLEANUP_INTERVAL"]:
                os.remove(file_path)
                logger.info(f"Removed old file: {file_path}")
    except Exception as e:
        logger.error(f"Output folder cleanup failed: {str(e)}")

@retry(tries=5, delay=5, backoff=2, exceptions=(Exception,))
async def face_swap(
    source_image: str,
    dest_image: str,
    dest_face_idx: int = 1,
    task_id: str = None
) -> str:
    client = None
    temp_output_path = None
    try:
        logger.info(f"Starting face swap for task {task_id} with source: {source_image}, dest: {dest_image}")
        if not all([validate_image(source_image), validate_image(dest_image)]):
            logger.error(f"Invalid input files: {source_image}, {dest_image}")
            raise ValueError("Invalid input files")

        progress_tracker[task_id] = "Initializing face swap"
        logger.debug(f"Initializing Gradio client for task {task_id}")
        try:
            client = Client("Dentro/face-swap", httpx_kwargs={"timeout": CONFIG["GRADIO_TIMEOUT"]})
            logger.debug(f"Gradio client initialized for task {task_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Gradio client: {str(e)}")
            raise HTTPException(500, detail=f"Gradio client initialization failed: {str(e)}")

        progress_tracker[task_id] = "Detecting faces and processing swap"
        logger.debug(f"Starting face swap with dest_face_idx: {dest_face_idx}")

        logger.info(f"Trying face swap with destination face number {dest_face_idx}")
        progress_tracker[task_id] = f"Trying face swap with destination face number {dest_face_idx}"
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_output:
                temp_output_path = temp_output.name
            
            result = client.predict(
                sourceImage=handle_file(source_image),
                sourceFaceIndex=1,
                destinationImage=handle_file(dest_image),
                destinationFaceIndex=dest_face_idx,
                api_name="/predict"
            )

            logger.info(f"Gradio face swap returned: {result}")
            if result and os.path.exists(result) and validate_image(result):
                shutil.copy(result, temp_output_path)
                logger.info(f"Face swap succeeded, copied to {temp_output_path}")
                progress_tracker[task_id] = f"Face swap succeeded"
            else:
                logger.error(f"Face swap failed or produced invalid result: {result}")
                progress_tracker[task_id] = f"Face swap failed"
                raise ValueError(f"Face swap failed for destination face index {dest_face_idx}")
        except Exception as e:
            logger.error(f"Face swap attempt failed: {str(e)}")
            progress_tracker[task_id] = f"Face swap failed: {str(e)}"
            raise

        progress_tracker[task_id] = "Applying tile-based upscaling"
        logger.debug(f"Enhancing face swap result with Tile-Upscaler")
        enhanced_output_url = await high_quality_enhance(temp_output_path)
        
        logger.info(f"Completed face swap (URL: {enhanced_output_url})")
        progress_tracker[task_id] = f"Completed face swap (URL: {enhanced_output_url})"
        return enhanced_output_url

    except Exception as e:
        progress_tracker[task_id] = f"Error: {str(e)}"
        logger.error(f"Face swap failed: {str(e)}")
        raise
    finally:
        if temp_output_path and os.path.exists(temp_output_path):
            try:
                os.unlink(temp_output_path)
                logger.debug(f"Cleaned up temporary file: {temp_output_path}")
            except FileNotFoundError:
                logger.warning(f"Temporary file {temp_output_path} already deleted")
        if client:
            try:
                client.close()
                logger.debug(f"Gradio client closed for task {task_id}")
            except Exception as e:
                logger.warning(f"Failed to close Gradio client: {str(e)}")

@app.options("/shopify-face-swap", description="Handle CORS preflight requests")
async def cors_preflight():
    return JSONResponse(
        status_code=200,
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400",
        }
    )

@app.get("/", description="Render the index page for the face-swap UI")
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result_image": None, "version": app.version}
    )

@app.post("/swap", description="Upload images for high-quality face-swapping")
async def swap_faces(
    source_image: UploadFile = File(...),
    dest_image: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    start_time = time.time()
    task_id = str(uuid.uuid4())
    progress_tracker[task_id] = "Starting"
    logger.info(f"Starting high-quality face swap task: {task_id}")

    try:
        if not (source_image.filename and dest_image.filename):
            logger.error("No file selected")
            raise HTTPException(400, detail="No file selected")
        
        if not (allowed_file(source_image.filename) and allowed_file(dest_image.filename)):
            logger.error(f"Invalid file format: {source_image.filename}, {dest_image.filename}")
            raise HTTPException(400, detail="Invalid file format. Only PNG, JPG, JPEG, WEBP allowed")

        source_content = await source_image.read()
        dest_content = await dest_image.read()
        if len(source_content) > CONFIG["MAX_FILE_SIZE"] or len(dest_content) > CONFIG["MAX_FILE_SIZE"]:
            logger.error(f"File size exceeds limit: {len(source_content)} or {len(dest_content)}")
            raise HTTPException(
                400,
                detail=f"File size exceeds {CONFIG['MAX_FILE_SIZE'] / (1024 * 1024)}MB"
            )

        cache_key = f"{get_file_hash(source_content)}:{get_file_hash(dest_content)}"

        with tempfile.TemporaryDirectory() as temp_dir:
            source_filename = f"source_{uuid.uuid4().hex}.{source_image.filename.rsplit('.', 1)[1]}"
            dest_filename = f"dest_{uuid.uuid4().hex}.{dest_image.filename.rsplit('.', 1)[1]}"
            source_path = os.path.join(temp_dir, source_filename)
            dest_path = os.path.join(temp_dir, dest_filename)

            with open(source_path, "wb") as f:
                f.write(source_content)
            with open(dest_path, "wb") as f:
                f.write(dest_content)

            result_url = await face_swap(source_path, dest_path, task_id=task_id)
            cache[cache_key] = result_url
            logger.info(f"Cached high-quality result: {result_url}")
            logger.debug(f"Returning result URL: {result_url}")

            background_tasks.add_task(cleanup_output_folder)

            return JSONResponse({
                "success": True,
                "data": {"result_image": result_url, "task_id": task_id},
                "error": None
            }, headers={"X-Process-Time": f"{time.time() - start_time:.2f}"})

    except HTTPException as e:
        logger.error(f"Face swap error: {str(e)}")
        progress_tracker[task_id] = f"Error: {str(e)}"
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "data": None, "error": str(e.detail)},
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        logger.error(f"Face swap error: {str(e)}")
        progress_tracker[task_id] = f"Error: {str(e)}"
        return JSONResponse(
            status_code=500,
            content={"success": False, "data": None, "error": str(e)},
            headers={"Access-Control-Allow-Origin": "*"}
        )

@app.get("/progress/{task_id}", description="Check progress of face swap task")
async def get_progress(task_id: str):
    status = progress_tracker.get(task_id, "Unknown task")
    return JSONResponse(
        content={"task_id": task_id, "status": status},
        headers={"Access-Control-Allow-Origin": "*"}
    )

@app.post("/shopify-face-swap", description="Upload user face and product image URL for Shopify preview")
async def shopify_face_swap(
    user_image: UploadFile = File(...),
    product_image_url: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    start_time = time.time()
    task_id = str(uuid.uuid4())
    progress_tracker[task_id] = "Starting"
    logger.info(f"Starting Shopify face swap task: {task_id}")

    temp_file_path = None
    try:
        if not user_image.filename:
            logger.error("No user image provided")
            raise HTTPException(400, detail="User image is required")
        if not product_image_url:
            logger.error("No product image URL provided")
            raise HTTPException(400, detail="Product image URL is required")
        if not product_image_url.startswith(('http://', 'https://')):
            logger.error(f"Invalid product image URL: {product_image_url}")
            raise HTTPException(400, detail="Invalid product image URL: Must start with http:// or https://")

        user_content = await user_image.read()
        if len(user_content) > CONFIG["MAX_FILE_SIZE"]:
            logger.error(f"User image size {len(user_content)} exceeds {CONFIG['MAX_FILE_SIZE']}")
            raise HTTPException(
                400,
                detail=f"User image size exceeds {CONFIG['MAX_FILE_SIZE'] / (1024 * 1024)}MB"
            )
        if not allowed_file(user_image.filename):
            logger.error(f"Invalid user image format: {user_image.filename}")
            raise HTTPException(400, detail="Invalid user image format. Only PNG, JPG, JPEG, WEBP allowed")

        progress_tracker[task_id] = "Downloading product image"
        try:
            response = requests.get(product_image_url, timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to download product image: HTTP {response.status_code}")
                raise HTTPException(400, detail=f"Failed to download product image: HTTP {response.status_code}")
            product_content = response.content
            content_type = response.headers.get("content-type", "")
            logger.info(f"Product image content-type: {content_type}")
            if not content_type.startswith("image/"):
                logger.error(f"Product image is not an image: Content-Type {content_type}")
                raise HTTPException(400, detail=f"Product image is not an image: Content-Type {content_type}")
        except requests.RequestException as e:
            logger.error(f"Product image download failed: {str(e)}")
            raise HTTPException(400, detail=f"Failed to download product image: {str(e)}")

        if len(product_content) > CONFIG["MAX_FILE_SIZE"]:
            logger.error(f"Product image size {len(product_content)} exceeds {CONFIG['MAX_FILE_SIZE']}")
            raise HTTPException(
                400,
                detail=f"Product image size exceeds {CONFIG['MAX_FILE_SIZE'] / (1024 * 1024)}MB"
            )

        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as temp_file:
            temp_file.write(product_content)
            temp_file_path = temp_file.name
        if not validate_image(temp_file_path):
            logger.error(f"Invalid product image format: {product_image_url}")
            raise HTTPException(400, detail=f"Invalid product image format: {product_image_url}")

        dest_face_idx = 1
        if "TEACHER.webp" in product_image_url:
            dest_face_idx = 3
        elif "REDKNIGHT.webp" in product_image_url:
            dest_face_idx = 4
        elif any(filename in product_image_url for filename in ["DOCTOR.webp", "BOYCHEF1FINAL.webp", "police_investigator.webp", "CULINARY_GIRL.png", "fsoccer.webp"]):
            dest_face_idx = 2

        cache_key = f"{get_file_hash(user_content)}:{get_file_hash(product_content)}:{dest_face_idx}"

        with tempfile.TemporaryDirectory() as temp_dir:
            user_filename = f"user_{uuid.uuid4().hex}.{user_image.filename.rsplit('.', 1)[1]}"
            product_ext = get_image_extension(product_content)
            product_filename = f"product_{uuid.uuid4().hex}.{product_ext}"
            user_path = os.path.join(temp_dir, user_filename)
            product_path = os.path.join(temp_dir, product_filename)

            with open(user_path, "wb") as f:
                f.write(user_content)
            with open(product_path, "wb") as f:
                f.write(product_content)

            result_url = await face_swap(user_path, product_path, dest_face_idx=dest_face_idx, task_id=task_id)
            cache[cache_key] = result_url
            logger.info(f"Cached high-quality result: {result_url}")
            logger.debug(f"Returning result URL: {result_url}")

            background_tasks.add_task(cleanup_output_folder)

            return JSONResponse({
                "success": True,
                "data": {"result_image": result_url, "task_id": task_id},
                "error": None
            }, headers={"X-Process-Time": f"{time.time() - start_time:.2f}"})

    except HTTPException as e:
        logger.error(f"Shopify face swap error: {str(e)}")
        progress_tracker[task_id] = f"Error: {str(e)}"
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"Cleaned up temporary file: {temp_file_path}")
            except FileNotFoundError:
                logger.warning(f"Temporary file {temp_file_path} already deleted")
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "data": None, "error": str(e.detail)},
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        logger.error(f"Shopify face swap error: {str(e)}")
        progress_tracker[task_id] = f"Error: {str(e)}"
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"Cleaned up temporary file: {temp_file_path}")
            except FileNotFoundError:
                logger.warning(f"Temporary file {temp_file_path} already deleted")
        return JSONResponse(
            status_code=500,
            content={"success": False, "data": None, "error": str(e)},
            headers={"Access-Control-Allow-Origin": "*"}
        )
