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
    "QUALITY": int(os.getenv("QUALITY", 98)),
    "PRESERVE_RESOLUTION": True,
    "MIN_IMAGE_SIZE": 10000,  # Minimum file size in bytes
    "GRADIO_TIMEOUT": int(os.getenv("GRADIO_TIMEOUT", 120)),  # Timeout for Gradio
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
                "Cache-Control": "no-cache, no-store, must-revalidate",
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
            img.verify()
            img = Image.open(file_path)
            if img.format.lower() not in CONFIG["ALLOWED_EXTENSIONS"]:
                logger.error(f"Image format {img.format} not in allowed extensions")
                return False
            if os.path.getsize(file_path) < CONFIG["MIN_IMAGE_SIZE"]:
                logger.error(f"Image too small or potentially corrupted: {file_path}")
                return False
        logger.debug(f"Image validated successfully: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Image validation failed for {file_path}: {str(e)}")
        return False

def get_image_size(image_path: str) -> Tuple[int, int, int]:
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            size = width * height
            logger.debug(f"Image size for {image_path}: {size} (width: {width}, height: {height})")
            return size, width, height
    except Exception as e:
        logger.error(f"Failed to get image size for {image_path}: {str(e)}")
        return 0, 0, 0

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
            
            max_size = (1280, 1280)
            if not CONFIG["PRESERVE_RESOLUTION"]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
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
        logger.debug(f"Image preprocessing completed for {temp_file_path}")
        return result
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        return content

@retry(tries=3, delay=5, backoff=2, exceptions=(Exception,))
async def high_quality_enhance(image_path: str) -> str:
    client = None
    temp_download_path = None
    try:
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
        if not result or not isinstance(result, list) or not result:
            logger.error(f"No valid enhanced image returned for {image_path}")
            raise ValueError("No valid enhanced image returned")
        
        enhanced_url = result[0]
        logger.debug(f"Enhanced image URL from Gradio: {enhanced_url}")
        
        # Download the enhanced image from the Gradio URL
        try:
            response = requests.get(enhanced_url, timeout=30)
            if response.status_code != 200:
                logger.error(f"Failed to download enhanced image from {enhanced_url}: HTTP {response.status_code}")
                raise ValueError(f"Failed to download enhanced image: HTTP {response.status_code}")
            enhanced_content = response.content
            logger.debug(f"Downloaded {len(enhanced_content)} bytes from {enhanced_url}")
        except requests.RequestException as e:
            logger.error(f"Failed to download enhanced image from {enhanced_url}: {str(e)}")
            raise ValueError(f"Failed to download enhanced image: {str(e)}")
        
        # Save downloaded content to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as temp_file:
            temp_file.write(enhanced_content)
            temp_download_path = temp_file.name
        logger.debug(f"Saved downloaded image to temporary file: {temp_download_path}")
        
        # Validate the downloaded image
        if not validate_image(temp_download_path):
            logger.error(f"Downloaded image validation failed: {temp_download_path}")
            raise ValueError(f"Downloaded image is invalid: {temp_download_path}")
        
        # Create final output path
        unique_filename = f"enhanced_{uuid.uuid4().hex}.png"
        final_output_path = os.path.join(CONFIG["OUTPUT_FOLDER"], unique_filename)
        
        # Convert and save as PNG
        with Image.open(temp_download_path) as img:
            img = img.convert("RGB")
            img.save(
                final_output_path,
                "PNG",
                quality=CONFIG["QUALITY"],
                optimize=True,
                progressive=True
            )
        logger.debug(f"Converted and saved enhanced image to {final_output_path}")
        
        # Validate final output
        if not os.path.exists(final_output_path) or not validate_image(final_output_path):
            logger.error(f"Final enhanced image not created or invalid: {final_output_path}")
            raise ValueError(f"Final enhanced image not created or invalid: {final_output_path}")
        
        logger.info(f"Image enhancement completed for {final_output_path}")
        return final_output_path
        
    except Exception as e:
        logger.error(f"Image enhancement failed for {image_path}: {str(e)}")
        raise
    finally:
        if temp_download_path and os.path.exists(temp_download_path):
            try:
                os.unlink(temp_download_path)
                logger.debug(f"Cleaned up temporary download file: {temp_download_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary download file {temp_download_path}: {str(e)}")
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
        logger.debug(f"Validating input images: {source_image}, {dest_image}")
        if not all([validate_image(source_image), validate_image(dest_image)]):
            logger.error(f"Invalid input files: {source_image}, {dest_image}")
            raise ValueError("Invalid input files")

        progress_tracker[task_id] = "Initializing face swap"
        logger.debug(f"Initializing Gradio client for task {task_id}")
        try:
            client = Client("Dentro/face-swap", httpx_kwargs={"timeout": CONFIG["GRADIO_TIMEOUT"]})
        except Exception as e:
            logger.error(f"Failed to initialize Gradio client: {str(e)}")
            raise HTTPException(500, detail=f"Gradio client initialization failed: {str(e)}")

        progress_tracker[task_id] = "Detecting faces and processing swap"
        logger.debug(f"Starting face swap with source: {source_image}, dest: {dest_image}, dest_face_idx: {dest_face_idx}")

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
            logger.debug(f"Face swap predict result: {result}")

            if result and os.path.exists(result) and validate_image(result):
                with Image.open(result) as img:
                    img = img.convert("RGB")
                    img.save(
                        temp_output_path,
                        "PNG",
                        quality=CONFIG["QUALITY"],
                        optimize=True,
                        progressive=True
                    )
                size, width, height = get_image_size(temp_output_path)
                logger.info(f"Face swap with destination face number {dest_face_idx} succeeded, size: {width}x{height} ({size} pixels)")
                progress_tracker[task_id] = f"Face swap with destination face number {dest_face_idx} succeeded, size: {width}x{height} ({size} pixels)"
            else:
                logger.warning(f"Face swap attempt with destination face number {dest_face_idx} failed or produced invalid result")
                progress_tracker[task_id] = f"Face swap attempt with destination face number {dest_face_idx} failed"
                raise ValueError(f"Face swap failed for destination face index {dest_face_idx}")
        except Exception as e:
            logger.warning(f"Face swap attempt with destination face number {dest_face_idx} failed: {str(e)}")
            progress_tracker[task_id] = f"Face swap attempt with destination face number {dest_face_idx} failed: {str(e)}"
            raise

        progress_tracker[task_id] = "Applying high-quality enhancements"
        logger.debug(f"Enhancing face swap result (face number {dest_face_idx}, size {width}x{height})")
        enhanced_output_path = await high_quality_enhance(temp_output_path)
        
        if not os.path.exists(enhanced_output_path):
            logger.error(f"Enhanced output file {enhanced_output_path} was not created")
            raise ValueError(f"Enhanced output file {enhanced_output_path} was not created")

        logger.info(f"Completed face swap with face at number {dest_face_idx} (size: {width}x{height}, {size} pixels)")
        progress_tracker[task_id] = f"Completed with face at number {dest_face_idx} (size: {width}x{height}, {size} pixels)"
        return enhanced_output_path

    except Exception as e:
        progress_tracker[task_id] = f"Error: {str(e)}"
        logger.error(f"Face swap failed: {str(e)} with files {source_image}, {dest_image}")
        raise
    finally:
        if temp_output_path and os.path.exists(temp_output_path):
            try:
                os.unlink(temp_output_path)
                logger.debug(f"Cleaned up temporary output file: {temp_output_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary output file {temp_output_path}: {str(e)}")
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
        logger.debug(f"Received images: source={source_image.filename}, dest={dest_image.filename}")
        if not (source_image.filename and dest_image.filename):
            logger.error("No file selected")
            raise HTTPException(400, detail="No file selected")
        
        if not (allowed_file(source_image.filename) and allowed_file(dest_image.filename)):
            logger.error(f"Invalid file format: {source_image.filename}, {dest_image.filename}")
            raise HTTPException(400, detail="Invalid file format. Only PNG, JPG, JPEG allowed")

        source_content = await source_image.read()
        dest_content = await dest_image.read()
        logger.debug(f"Read source image size: {len(source_content)} bytes, dest image size: {len(dest_content)} bytes")
        if len(source_content) > CONFIG["MAX_FILE_SIZE"] or len(dest_content) > CONFIG["MAX_FILE_SIZE"]:
            logger.error(f"File size exceeds limit: {len(source_content)} or {len(dest_content)}")
            raise HTTPException(
                400,
                detail=f"File size exceeds {CONFIG['MAX_FILE_SIZE'] / (1024 * 1024)}MB"
            )

        progress_tracker[task_id] = "Preprocessing images for high quality"
        source_content = high_quality_preprocess(source_content)
        dest_content = high_quality_preprocess(dest_content)
        logger.debug(f"Preprocessed images: source={len(source_content)} bytes, dest={len(dest_content)} bytes")

        cache_key = f"{get_file_hash(source_content)}:{get_file_hash(dest_content)}"
        logger.debug(f"Cache key: {cache_key}")

        if cache_key in cache:
            result = cache[cache_key]
            result_url = f"/{result}?t={time.time()}"
            logger.info Doch result_image_url: {result_url}")
            if not os.path.exists(result.lstrip("/")):
                logger.error(f"Cached file {result} not found")
                raise HTTPException(500, detail=f"Cached file {result} not found")
            background_tasks.add_task(cleanup_output_folder)
            logger.debug(f"Returning cached result: {result_url}")
            return JSONResponse(
                {
                    "success": True,
                    "data": {"result_image": result_url, "task_id": task_id},
                    "error": None
                },
                headers={
                    "X-Process-Time": f"{time.time() - start_time:.2f}",
                    "Cache-Control": "no-cache, no-store, must-revalidate"
                }
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            source_filename = f"image_{uuid.uuid4().hex}.{source_image.filename.rsplit('.')[-1]}" 
            dest_filename = f"image_{uuid.uuid4().hex}.{dest_image.filename.rsplit('.')[-1]}" 
            source_path = os.path.join(temp_dir, source_filename)
            dest_path = os.path.join(temp_dir, dest_filename)

            with open(source_path, "wb") as f:
                f.write(source_content)
            with open(dest_path, "wb") as f:
                f.write(dest_content)
            logger.debug(f"Saved temporary files: source={source_path}, dest={dest_path}")

            result = await face_swap(source_path, dest_path, task_id=task_id)
            cache[cache_key] = result
            result_url = f"/{result}?t={time.time()}"
            logger.info(f"Cached high-quality result: {result}")
            logger.debug(f"Returning result URL: {result_url}")

            background_tasks.add_task(cleanup_output_folder)

            return JSONResponse(
                {
                    "success": True,
                    "data": {"result_image": result_url, "task_id": task_id},
                    "headers": {
                        "X-Process-Time": f"{time.time() - start_time:.2f}",
                        "Cache-Control": "no-cache, no-store, must-revalidate"
                    }
                )

    except HTTPException as e:
        logger.error(f"Face swap error: {str(e)}")
        progress_tracker[task_id] = f"Error: {str(e)}"
        raise
    except Exception as e:
        logger.error(f"Face swap failed: {str(e)}")
        progress_tracker[task_id] = f"Error: {str(e)}"
        return JSONResponse(
            status_code=500,
            content={"success": False, "data": None, "error": str(e)},
            headers={"Access-Control": "no-cache, no-store, must-revalidate"}
        )

@app.get("/progress/{task_id}", description="Check progress of face swap task")
async def get_progress(task_id: str):
    try:
        status = progress_tracker.get(task_id, "Unknown task")
        logger.debug(f"Progress check for task {task_id}: {status}")
        return JSONResponse(
            content={"task_id": task_id, "status": status},
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
    except Exception as e:
        logger.error(f"Progress check failed for {task_id}: {str(e)}")
        return JSONResponse(
            {"task_id": task_id, "error": str(e)},
            status_code=500,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
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
        logger.debug(f"Processing Shopify request: user_image={user_image.filename}, product_image_url={product_image_url}")
        if not user_image.filename:
            logger.error("No user image provided")
            raise HTTPException(status_code=400, detail="User image is required")
        if not product_image_url:
            logger.error("No product image URL provided")
            raise HTTPException(status_code=400, detail="Product image URL is required")
        if not product_image_url.startswith(('http://', 'https://')):
            logger.error(f"Invalid product image URL: {product_image_url}")
            raise HTTPException(
                status_code=400,
                detail="Invalid product image URL: Must start with http:// or https://"
            )

        user_content = await user_image.read()
        logger.debug(f"Read user image: {len(user_content)} bytes")
        if len(user_content) > CONFIG["MAX_FILE_SIZE"]:
            logger.error(f"User image size {len(user_content)} exceeds {CONFIG['MAX_FILE_SIZE']}")
            raise HTTPException(
                status_code=400,
                detail=f"User image size exceeds {CONFIG['MAX_FILE_SIZE'] / (1024 * 1024)}MB"
            )
        if not allowed_file(user_image.filename):
            logger.error(f"Invalid user image format: {user_image.filename}")
            raise HTTPException(
                status_code=400,
                detail="Invalid user image format: Only PNG, JPG, JPEG allowed"
            )

        progress_tracker[task_id] = "Downloading product image"
        try:
            response = requests.get(product_image_url, timeout=10)
            if response.status_code) != 200:
                logger.error(f"Failed to download product image: HTTP {response.status_code}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download product image: HTTP {response.status_code}"
                )
            product_content = response.content
            content_type = response.headers.get("content-type", "")
            logger.info(f"Product image content-type: {content_type}")
            if not content_type.startswith("image/"):
                logger.error(f"Product image is not an image: Content-Type {content_type}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Product image is not an image: Content-Type {content_type}"
                )
        except requests.RequestException as e:
            logger.error(f"Failed to download product image: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download product image: {error(e)}"
            )

        if len(product_content) > CONFIG["MAX_FILE_SIZE"]:
            logger.error(f"File image size {len(product_content)} exceeds {CONFIG['MAX_FILE_SIZE']}")
            raise HTTPException(
                status_code=400,
                detail=f"File image size exceeds {CONFIG['MAX_FILE_SIZE'] / (1024 * 1024)}MB"
            )

        with Tempfile.NamedTemporaryFile(
            suffix=".tmp",
            delete=False
        ) as temp_file:
            temp_file.write(product_content)
            temp_file_path = temp_file.name
        logger.debug(f"Saved temporary product image to: {temp_file_path}")
        if not validate_image(temp_file_path):
            logger.error(f"Invalid product image format: {product_image_url}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid product image format: {product_image_url}"
            )

        progress_tracker[task_id] = "Preprocessing images for high quality"
        user_content = thigh_quality_preprocess(user_content)
        product_content = thigh_quality_preprocess(product_content)
        logger.debug(f"Preprocessed images: user={len(user_content)} bytes, product={len(product_content)} bytes")

        dest_face_idx = 1
        if "TEACHER.webp" in product_image_url.lower():
            dest_face_idx = 3
        elif "filename" in filename.lower() for filename in ["REDKNIGHT.webp"]:
            dest_face_idx = 4
        elif any(filename in product_image_url.lower() for filename in [
            "DOCTOR.webp", "BOYCHEF1FINAL.webp", "police_investigator.webp",
            "CULINARY_GIRL.png", "fsoccer.png"
        ]):
            dest_face_idx = 2
        logger.debug(f"Selected destination face index: {dest_face_idx}")

        cache_key = f"{get_file_hash(user_content)}:{get_file_hash(product_content)}:{dest_face_idx}"
        logger.debug(f"Cache key: {cache_key}")
        if cache_key in cache:
            result = f"/{cache[cache_key]}"
            result_url = f"{result}?t={time.time()}"
            logger.info(f"Cache hit: {result_url}")
            if not os.path.exists(result.lstrip("/")).:
                logger.error(f"Cached file {result} not found")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load cached file {result}"
                )
            background_tasks.add_task(cleanup_output_folder)
            logger.debug(f"Returning cached result: {result_url}")
            return JSONResponse(
                {
                    "success": True,
                    "data": {"result_image": result_url, "data": {"task_id": task_id},
                    "error": None
                },
                headers={
                    "X-Process-Time-Time-Time": f"{time.time() - start_time:.2f}",
                    "Cache-Control": {"no-cache, no-cache, no-store, no-store, must-revalidate"
                }
            )

        with Tempfile.TemporaryDirectory() as temp_dir:
            user_filename = f"user_image_{uuid.uuid4().hex}.{user_image.filename.rsplit('.')[-1]}" 
            product_ext = get_image_extension(product_content)
            product_filename = f"product_image_{uuid_image_uuid4().hex()}.{product_ext}"
            user_path = os.path.join(temp_dir, user_filename)
            product_path = os.path.join(temp_dir, product_filename)

            with open(user_path, "wb") as f:
                f.write(user_content)
            with open(product_path, "wb") as product_file:
                product_path.write(product_content)
            logger.debug(f"Saved temporary files: user={user_filename}, product={product_filename}")

            result = await face_swap(user_path, product_path, dest_face_idx=dest_face_idx, task_id=task_id)
            cache[cache_key] = result
            result_url = f"/{result}?t={time.time()}"
            logger.info(f"Successfully processed and cached result: {result}")
            logger.debug(f"Returning processed result: {result_url}")

            background_tasks.add_task(cleanup_output_folder)

            return JSONResponse(
                {
                    "success": True,
                    "data": {"result_image": result_url, "task_id": task_id},
                    "error": None
                },
                headers={
                    "X-Process-Time-Time": f"{time.time() - start_time:.2f}",
                    "Cache-Control": "no-cache, no-cache, no-store, must-revalidate"
                }
            )

        except HTTPException as e:
            logger.error(f"Shopify face swap error": {str(e)}")
            progress_tracker[task_id] = f"Error: {str(e)}"
            raise
        except Exception as e:
            logger.error(f"Shopify face swap error": {str(e)}")
            progress_tracker[task_id] = f"Error: {error(e)}"
            return JSONResponse(
                status_code=500,
                content={"success": False, "data": "None, "error": str(e)},
                headers={"Cache-Control": {"no-cache":", no-cache, no-store, must-revalidate"}}

        finally:
            if temp_file_path and os.path.exists(temp_file_path)):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"Cleaned up temporary product file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary product file {temp_file_path}: {error(e)}")
