import os
import uuid
import hashlib
import tempfile
import time
import logging
import requests
from pathlib import Path
from typing import Optional, Dict
from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks, HTTPException, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance, ImageFilter
from cachetools import TTLCache
from gradio_client import Client, handle_file
from retry import retry
import asyncio

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
    allow_origins=["*"],  # Allow all origins
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
    "CLEANUP_INTERVAL": int(os.getenv("CLEANUP_INTERVAL", 3600)),
    "QUALITY": int(os.getenv("QUALITY", 98)),
    "PRESERVE_RESOLUTION": True
}

# Create directories
for folder in [CONFIG["STATIC_DIR"], CONFIG["UPLOAD_FOLDER"], CONFIG["OUTPUT_FOLDER"]]:
    os.makedirs(folder, exist_ok=True)

# Custom StaticFiles
class CORSStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers.update({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Cache-Control": "public, max-age=3600",
            "Content-Disposition": "inline"
        })
        return response

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
            img.verify()
        return file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    except Exception as e:
        logger.error(f"Image validation failed for {file_path}: {str(e)}")
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

def high_quality_preprocess(content: bytes) -> bytes:
    try:
        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        with Image.open(temp_file_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if CONFIG["PRESERVE_RESOLUTION"]:
                img = img.copy()
            else:
                original_size = img.size
                img.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
                if img.size != original_size:
                    img = img.resize(original_size, Image.Resampling.LANCZOS)
            
            img = ImageEnhance.Color(img).enhance(1.05)
            img = ImageEnhance.Contrast(img).enhance(1.02)
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as output_file:
                img.save(
                    output_file.name,
                    "PNG",
                    optimize=True,
                    quality=CONFIG["QUALITY"],
                    progressive=True
                )
                with open(output_file.name, "rb") as f:
                    result = f.read()
        
        os.unlink(temp_file_path)
        os.unlink(output_file.name)
        return result
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        return content

def high_quality_enhance(image_path: str, enhancements: Dict[str, float] = None) -> None:
    try:
        enhancements = enhancements or {
            "sharpness": 1.15,
            "contrast": 1.02,
            "brightness": 1.03,
            "color": 1.05
        }

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = ImageEnhance.Sharpness(img).enhance(enhancements["sharpness"])
            img = ImageEnhance.Contrast(img).enhance(enhancements["contrast"])
            img = ImageEnhance.Brightness(img).enhance(enhancements["brightness"])
            img = ImageEnhance.Color(img).enhance(enhancements["color"])
            img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=100, threshold=3))
            img.save(
                image_path,
                "PNG",
                optimize=True,
                quality=CONFIG["QUALITY"],
                progressive=True
            )
    except Exception as e:
        logger.error(f"Image enhancement failed: {str(e)}")

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

@retry(tries=5, delay=2, backoff=2, exceptions=(Exception,))
async def face_swap(
    source_image: str,
    dest_image: str,
    source_face_idx: int = 1,
    dest_face_idx: int = 1,
    task_id: str = None
) -> str:
    try:
        if not all([validate_image(source_image), validate_image(dest_image)]):
            raise ValueError("Invalid input files")

        progress_tracker[task_id] = "Initializing face swap"
        try:
            client = Client("Dentro/face-swap")
        except Exception as e:
            logger.error(f"Failed to initialize Gradio client: {str(e)}")
            raise HTTPException(500, detail=f"Gradio client initialization failed: {str(e)}")

        progress_tracker[task_id] = "Processing high-quality face swap"
        result = client.predict(
            sourceImage=handle_file(source_image),
            sourceFaceIndex=source_face_idx,
            destinationImage=handle_file(dest_image),
            destinationFaceIndex=dest_face_idx,
            api_name="/predict"
        )

        if not result or not os.path.exists(result):
            logger.warning(f"Initial face swap attempt failed with indices {source_face_idx}, {dest_face_idx}")
            for idx in [0, 2]:
                if source_face_idx != idx:
                    result = client.predict(
                        sourceImage=handle_file(source_image),
                        sourceFaceIndex=idx,
                        destinationImage=handle_file(dest_image),
                        destinationFaceIndex=dest_face_idx,
                        api_name="/predict"
                    )
                    if result and os.path.exists(result):
                        logger.info(f"Successful fallback with source index {idx}")
                        break
            if not result or not os.path.exists(result):
                raise ValueError("All face swap attempts failed")

        unique_filename = f"face_swap_{uuid.uuid4().hex}.png"
        output_path = os.path.join(CONFIG["OUTPUT_FOLDER"], unique_filename)
        
        progress_tracker[task_id] = "Applying high-quality enhancements"
        with Image.open(result) as img:
            img = img.convert("RGB")
            img.save(
                output_path,
                "PNG",
                quality=CONFIG["QUALITY"],
                optimize=True,
                progressive=True
            )
        high_quality_enhance(output_path)
        
        progress_tracker[task_id] = "Completed"
        return output_path
    except Exception as e:
        progress_tracker[task_id] = f"Error: {str(e)}"
        logger.error(f"Face swap failed: {str(e)} with files {source_image}, {dest_image}")
        raise

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
    source_face_idx: int = 1,
    dest_face_idx: int = 1,
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

        progress_tracker[task_id] = "Preprocessing images for high quality"
        source_content = high_quality_preprocess(source_content)
        dest_content = high_quality_preprocess(dest_content)

        cache_key = f"{get_file_hash(source_content)}:{get_file_hash(dest_content)}:{source_face_idx}:{dest_face_idx}"

        if cache_key in cache:
            result_url = f"/{cache[cache_key]}"
            logger.info(f"Cache hit: {result_url}")
            background_tasks.add_task(cleanup_output_folder)
            return JSONResponse({
                "success": True,
                "data": {"result_image": result_url, "task_id": task_id},
                "error": None
            }, headers={"X-Process-Time": f"{time.time() - start_time:.2f}"})

        with tempfile.TemporaryDirectory() as temp_dir:
            source_filename = f"source_{uuid.uuid4().hex}.{source_image.filename.rsplit('.', 1)[1]}"
            dest_filename = f"dest_{uuid.uuid4().hex}.{dest_image.filename.rsplit('.', 1)[1]}"
            source_path = os.path.join(temp_dir, source_filename)
            dest_path = os.path.join(temp_dir, dest_filename)

            with open(source_path, "wb") as f:
                f.write(source_content)
            with open(dest_path, "wb") as f:
                f.write(dest_content)

            result = await face_swap(source_path, dest_path, source_face_idx, dest_face_idx, task_id)
            cache[cache_key] = result
            logger.info(f"Cached high-quality result: {result}")

            background_tasks.add_task(cleanup_output_folder)

            return JSONResponse({
                "success": True,
                "data": {"result_image": f"/{result}", "task_id": task_id},
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
            os.unlink(temp_file_path)
            logger.error(f"Invalid product image format: {product_image_url}")
            raise HTTPException(400, detail="Invalid product image format. Only PNG, JPG, JPEG, WEBP allowed")

        progress_tracker[task_id] = "Preprocessing images for high quality"
        user_content = high_quality_preprocess(user_content)
        product_content = high_quality_preprocess(product_content)

        cache_key = f"{get_file_hash(user_content)}:{get_file_hash(product_content)}:1:1"
        if cache_key in cache:
            result_url = f"/{cache[cache_key]}"
            logger.info(f"Cache hit: {result_url}")
            background_tasks.add_task(cleanup_output_folder)
            os.unlink(temp_file_path)
            return JSONResponse({
                "success": True,
                "data": {"result_image": result_url, "task_id": task_id},
                "error": None
            }, headers={"X-Process-Time": f"{time.time() - start_time:.2f}"})

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

            result = await face_swap(user_path, product_path, 1, 1, task_id)
            cache[cache_key] = result
            logger.info(f"Cached high-quality result: {result}")

        background_tasks.add_task(cleanup_output_folder)
        os.unlink(temp_file_path)

        return JSONResponse({
            "success": True,
            "data": {"result_image": f"/{result}", "task_id": task_id},
            "error": None
        }, headers={"X-Process-Time": f"{time.time() - start_time:.2f}"})

    except HTTPException as e:
        logger.error(f"Shopify face swap error: {str(e)}")
        progress_tracker[task_id] = f"Error: {str(e)}"
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "data": None, "error": str(e.detail)},
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        logger.error(f"Shopify face swap error: {str(e)}")
        progress_tracker[task_id] = f"Error: {str(e)}"
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
        return JSONResponse(
            status_code=500,
            content={"success": False, "data": None, "error": str(e)},
            headers={"Access-Control-Allow-Origin": "*"}
        )
