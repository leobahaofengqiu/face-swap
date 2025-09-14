import os
import uuid
import hashlib
import tempfile
import time
import logging
import requests
import json
from datetime import datetime
import pytz
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks, HTTPException, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from cachetools import TTLCache
from gradio_client import Client, handle_file
from retry import retry
import asyncio
import shutil
import sqlite3
from dotenv import load_dotenv
from logging import Formatter

# Load environment variables
load_dotenv()

# Configure logging with China Standard Time
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
cst_tz = pytz.timezone('Asia/Shanghai')
formatter = Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %Z'
)
formatter.converter = lambda *args: datetime.now(cst_tz).timetuple()
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

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

# Configuration for Railway
CONFIG = {
    "UPLOAD_FOLDER": os.getenv("UPLOAD_FOLDER", "static/uploads"),
    "OUTPUT_FOLDER": os.getenv("OUTPUT_FOLDER", "static/output"),
    "DATA_FOLDER": os.getenv("DATA_FOLDER", "data"),
    "STATIC_DIR": "static",
    "ALLOWED_EXTENSIONS": {'png', 'jpg', 'jpeg', 'webp'},
    "MAX_FILE_SIZE": int(os.getenv("MAX_FILE_SIZE", 30 * 1024 * 1024)),  # 30MB
    "CACHE_TTL": int(os.getenv("CACHE_TTL", 7200)),  # 2 hours
    "MAX_CACHE_SIZE": int(os.getenv("MAX_CACHE_SIZE", 100)),
    "CLEANUP_INTERVAL": int(os.getenv("CLEANUP_INTERVAL", 86400)),  # 24 hours
    "MIN_IMAGE_SIZE": 10000,
    "GRADIO_TIMEOUT": int(os.getenv("GRADIO_TIMEOUT", 120)),
    "HF_TOKEN": os.getenv("HF_TOKEN"),
    "PORT": int(os.getenv("PORT", 8000)),  # Railway port
    "DATABASE_URL": os.getenv("DATABASE_URL", "sqlite:///./face_swap_data.db")
}

# Database setup
def init_database():
    """Initialize SQLite database for storing face swap data"""
    db_path = "face_swap_data.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_swap_records (
            id TEXT PRIMARY KEY,
            task_id TEXT UNIQUE NOT NULL,
            source_image_path TEXT,
            target_image_path TEXT,
            face_swap_output_path TEXT,
            enhanced_output_path TEXT,
            source_image_hash TEXT,
            target_image_hash TEXT,
            dest_face_idx INTEGER DEFAULT 1,
            processing_time REAL,
            status TEXT DEFAULT 'pending',
            error_message TEXT,
            created_at TEXT,
            updated_at TEXT,
            metadata TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

# Initialize database on startup
init_database()

def save_face_swap_record(task_id: str, data: Dict):
    """Save face swap record to database with CST timestamps"""
    try:
        conn = sqlite3.connect("face_swap_data.db")
        cursor = conn.cursor()
        cst_tz = pytz.timezone('Asia/Shanghai')
        current_time = datetime.now(cst_tz).strftime('%Y-%m-%d %H:%M:%S')
        
        cursor.execute("""
            INSERT OR REPLACE INTO face_swap_records 
            (id, task_id, source_image_path, target_image_path, face_swap_output_path, 
             enhanced_output_path, source_image_hash, target_image_hash, dest_face_idx,
             processing_time, status, error_message, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            task_id,
            data.get('source_image_path'),
            data.get('target_image_path'),
            data.get('face_swap_output_path'),
            data.get('enhanced_output_path'),
            data.get('source_image_hash'),
            data.get('target_image_hash'),
            data.get('dest_face_idx', 1),
            data.get('processing_time'),
            data.get('status', 'completed'),
            data.get('error_message'),
            json.dumps(data.get('metadata', {})),
            current_time,
            current_time
        ))
        conn.commit()
        conn.close()
        logger.info(f"Face swap record saved for task: {task_id}")
    except Exception as e:
        logger.error(f"Failed to save face swap record: {str(e)}")

def get_face_swap_records(limit: int = 100):
    """Get face swap records from database"""
    try:
        conn = sqlite3.connect("face_swap_data.db")
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM face_swap_records 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        records = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return records
    except Exception as e:
        logger.error(f"Failed to get face swap records: {str(e)}")
        return []

# Verify HF_TOKEN exists
if not CONFIG["HF_TOKEN"]:
    logger.error("Hugging Face token (HF_TOKEN) not found in environment variables")
    raise Exception("Hugging Face token (HF_TOKEN) is required")

# Create directories and verify permissions
for folder in [CONFIG["STATIC_DIR"], CONFIG["UPLOAD_FOLDER"], CONFIG["OUTPUT_FOLDER"], CONFIG["DATA_FOLDER"]]:
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

# Cache setup
cache = TTLCache(maxsize=CONFIG["MAX_CACHE_SIZE"], ttl=CONFIG["CACHE_TTL"])

# Progress tracking with URLs
progress_tracker: Dict[str, Dict[str, str]] = {}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in CONFIG["ALLOWED_EXTENSIONS"]

def validate_image(file_path: str) -> bool:
    try:
        with Image.open(file_path) as img:
            img.verify()
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
        headers = {"Authorization": f"Bearer {CONFIG['HF_TOKEN']}"}
        response = requests.head(url, headers=headers, timeout=5)
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

def save_permanent_image(content: bytes, filename: str, folder: str) -> str:
    """Save image permanently and return relative path"""
    try:
        file_path = os.path.join(folder, filename)
        with open(file_path, "wb") as f:
            f.write(content)
        return file_path
    except Exception as e:
        logger.error(f"Failed to save permanent image {filename}: {str(e)}")
        return None

async def high_quality_enhance(image_path: str, task_id: str, dest_image_path: str) -> Tuple[str, str]:
    codeformer_client = None
    face_swap_url = None
    codeformer_url = None
    try:
        logger.info(f"Starting enhancement pipeline for {image_path} (task {task_id})")
        
        # Save face swap result
        face_swap_filename = f"face_swap_{task_id}_{uuid.uuid4().hex}.png"
        face_swap_path = os.path.join(CONFIG["OUTPUT_FOLDER"], face_swap_filename)
        shutil.copy(image_path, face_swap_path)
        face_swap_url = f"/static/output/{face_swap_filename}"
        logger.info(f"Face swap image saved at: {face_swap_url}")
        progress_tracker[task_id]["face_swap_url"] = face_swap_url

        # Enhance with CodeFormer
        progress_tracker[task_id]["status"] = "Enhancing with CodeFormer"
        logger.debug(f"Initializing CodeFormer client for {image_path}")
        codeformer_client = Client(
            "sczhou/CodeFormer",
            hf_token=CONFIG["HF_TOKEN"],
            httpx_kwargs={"timeout": CONFIG["GRADIO_TIMEOUT"]}
        )
        
        logger.debug(f"Sending image {image_path} to CodeFormer")
        codeformer_result = codeformer_client.predict(
            image=handle_file(image_path),
            face_align=True,
            background_enhance=True,
            face_upsample=True,
            upscale=1,
            codeformer_fidelity=0.8,
            api_name="/predict"
        )
        
        logger.debug(f"CodeFormer result: {codeformer_result}")
        if not codeformer_result:
            logger.error(f"No valid output from CodeFormer for {image_path}")
            raise ValueError("No valid output from CodeFormer")
        
        # Get original dimensions
        with Image.open(dest_image_path) as dest_img:
            original_width, original_height = dest_img.size

        # Handle CodeFormer result
        if codeformer_result.startswith(('http://', 'https://')):
            if not validate_image_url(codeformer_result):
                logger.error(f"Invalid or inaccessible CodeFormer image URL: {codeformer_result}")
                raise ValueError(f"Invalid or inaccessible CodeFormer image URL: {codeformer_result}")
            
            response = requests.get(codeformer_result, headers={"Authorization": f"Bearer {CONFIG['HF_TOKEN']}"}, timeout=10)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            
            with Image.open(temp_file_path) as img:
                img = img.resize((original_width, original_height), Image.Resampling.LANCZOS)
                output_filename = f"enhanced_{task_id}_{uuid.uuid4().hex}.png"
                output_path = os.path.join(CONFIG["OUTPUT_FOLDER"], output_filename)
                img.save(output_path, format="PNG")
            
            os.unlink(temp_file_path)
            codeformer_url = f"/static/output/{output_filename}"
            logger.info(f"CodeFormer enhanced image saved at: {codeformer_url}")
            progress_tracker[task_id]["codeformer_url"] = codeformer_url
            return face_swap_url, codeformer_url
        else:
            if not os.path.exists(codeformer_result) or not validate_image(codeformer_result):
                logger.error(f"Invalid CodeFormer output: {codeformer_result}")
                raise ValueError("Invalid CodeFormer output")
            
            with Image.open(codeformer_result) as img:
                img = img.resize((original_width, original_height), Image.Resampling.LANCZOS)
                output_filename = f"enhanced_{task_id}_{uuid.uuid4().hex}.png"
                output_path = os.path.join(CONFIG["OUTPUT_FOLDER"], output_filename)
                img.save(output_path, format="PNG")
            
            codeformer_url = f"/static/output/{output_filename}"
            logger.info(f"CodeFormer enhanced image saved at: {codeformer_url}")
            progress_tracker[task_id]["codeformer_url"] = codeformer_url
            return face_swap_url, codeformer_url
        
    except Exception as e:
        logger.error(f"Image enhancement pipeline failed for {image_path}: {str(e)}")
        progress_tracker[task_id]["status"] = f"Error: {str(e)}"
        raise
    finally:
        if codeformer_client:
            try:
                codeformer_client.close()
                logger.debug(f"CodeFormer client closed for {image_path}")
            except Exception as e:
                logger.warning(f"Failed to close CodeFormer client: {str(e)}")

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
    task_id: str = None,
    source_content: bytes = None,
    dest_content: bytes = None
) -> Tuple[str, Dict]:
    client = None
    temp_output_path = None
    start_time = time.time()
    
    try:
        logger.info(f"Starting face swap for task {task_id} with source: {source_image}, dest: {dest_image}")
        progress_tracker[task_id] = {"status": "Starting", "face_swap_url": None, "codeformer_url": None}
        
        if not all([validate_image(source_image), validate_image(dest_image)]):
            logger.error(f"Invalid input files: {source_image}, {dest_image}")
            raise ValueError("Invalid input files")

        # Save permanent copies of source and target images
        source_hash = get_file_hash(source_content) if source_content else None
        dest_hash = get_file_hash(dest_content) if dest_content else None
        
        source_filename = f"source_{task_id}_{uuid.uuid4().hex}.png"
        dest_filename = f"dest_{task_id}_{uuid.uuid4().hex}.png"
        
        permanent_source_path = None
        permanent_dest_path = None
        
        if source_content:
            permanent_source_path = save_permanent_image(source_content, source_filename, CONFIG["DATA_FOLDER"])
        if dest_content:
            permanent_dest_path = save_permanent_image(dest_content, dest_filename, CONFIG["DATA_FOLDER"])

        progress_tracker[task_id]["status"] = "Initializing face swap"
        logger.debug(f"Initializing Gradio client for task {task_id}")
        try:
            client = Client(
                "Dentro/face-swap",
                hf_token=CONFIG["HF_TOKEN"],
                httpx_kwargs={"timeout": CONFIG["GRADIO_TIMEOUT"]}
            )
            logger.debug(f"Gradio client initialized for task {task_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Gradio client: {str(e)}")
            raise HTTPException(500, detail=f"Gradio client initialization failed: {str(e)}")

        progress_tracker[task_id]["status"] = "Detecting faces and processing swap"
        logger.debug(f"Starting face swap with dest_face_idx: {dest_face_idx}")

        logger.info(f"Trying face swap with destination face number {dest_face_idx}")
        progress_tracker[task_id]["status"] = f"Trying face swap with destination face number {dest_face_idx}"
        
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
                
                face_swap_filename = f"face_swap_{task_id}_{uuid.uuid4().hex}.png"
                face_swap_path = os.path.join(CONFIG["OUTPUT_FOLDER"], face_swap_filename)
                shutil.copy(result, face_swap_path)
                face_swap_url = f"/static/output/{face_swap_filename}"
                logger.info(f"Face swap image saved at: {face_swap_url}")
                progress_tracker[task_id]["face_swap_url"] = face_swap_url
                progress_tracker[task_id]["status"] = f"Face swap succeeded (URL: {face_swap_url})"
            else:
                logger.error(f"Face swap failed or produced invalid result: {result}")
                progress_tracker[task_id]["status"] = f"Face swap failed"
                raise ValueError(f"Face swap failed for destination face index {dest_face_idx}")
                
        except Exception as e:
            logger.error(f"Face swap attempt failed: {str(e)}")
            progress_tracker[task_id]["status"] = f"Face swap failed: {str(e)}"
            raise

        progress_tracker[task_id]["status"] = "Applying enhancement pipeline"
        logger.debug(f"Enhancing face swap result with CodeFormer")
        face_swap_url, codeformer_url = await high_quality_enhance(temp_output_path, task_id, dest_image)
        
        processing_time = time.time() - start_time
        
        # Save data to database
        face_swap_data = {
            'source_image_path': permanent_source_path,
            'target_image_path': permanent_dest_path,
            'face_swap_output_path': face_swap_url,
            'enhanced_output_path': codeformer_url,
            'source_image_hash': source_hash,
            'target_image_hash': dest_hash,
            'dest_face_idx': dest_face_idx,
            'processing_time': processing_time,
            'status': 'completed',
            'metadata': {
                'gradio_result': result,
                'enhancement_applied': True,
                'original_dimensions': None
            }
        }
        
        save_face_swap_record(task_id, face_swap_data)
        
        logger.info(f"Completed face swap (Final URL: {codeformer_url})")
        progress_tracker[task_id]["status"] = f"Completed face swap (Final URL: {codeformer_url})"
        
        return codeformer_url, face_swap_data

    except Exception as e:
        processing_time = time.time() - start_time
        progress_tracker[task_id]["status"] = f"Error: {str(e)}"
        logger.error(f"Face swap failed: {str(e)}")
        
        # Save error to database
        error_data = {
            'processing_time': processing_time,
            'status': 'failed',
            'error_message': str(e),
            'dest_face_idx': dest_face_idx
        }
        save_face_swap_record(task_id, error_data)
        
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

# New endpoints to serve source and target images
@app.get("/source-image/{task_id}")
async def get_source_image(task_id: str):
    """Serve the source image for a given task ID"""
    try:
        conn = sqlite3.connect("face_swap_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT source_image_path FROM face_swap_records WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        conn.close()
        if not row or not row[0]:
            raise HTTPException(404, detail="Source image not found")
        if not os.path.exists(row[0]):
            raise HTTPException(404, detail="Source image file missing")
        return FileResponse(
            row[0],
            media_type="image/png",
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        logger.error(f"Failed to serve source image: {str(e)}")
        raise HTTPException(500, detail=str(e))

@app.get("/target-image/{task_id}")
async def get_target_image(task_id: str):
    """Serve the target image for a given task ID"""
    try:
        conn = sqlite3.connect("face_swap_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT target_image_path FROM face_swap_records WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        conn.close()
        if not row or not row[0]:
            raise HTTPException(404, detail="Target image not found")
        if not os.path.exists(row[0]):
            raise HTTPException(404, detail="Target image file missing")
        return FileResponse(
            row[0],
            media_type="image/png",
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        logger.error(f"Failed to serve target image: {str(e)}")
        raise HTTPException(500, detail=str(e))

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    cst_tz = pytz.timezone('Asia/Shanghai')
    return {
        "status": "healthy",
        "timestamp": datetime.now(cst_tz).isoformat()
    }

@app.options("/shopify-face-swap")
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

@app.get("/")
async def index(request: Request, task_id: str = None):
    try:
        templates = Jinja2Templates(directory="templates")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result_image": None, "version": app.version, "task_id": task_id}
        )
    except Exception:
        return JSONResponse({"message": "Face Swap API is running", "version": app.version})

@app.post("/swap")
async def swap_faces(
    source_image: UploadFile = File(...),
    dest_image: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    start_time = time.time()
    task_id = str(uuid.uuid4())
    progress_tracker[task_id] = {"status": "Starting", "face_swap_url": None, "codeformer_url": None}
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

        with tempfile.TemporaryDirectory() as temp_dir:
            source_filename = f"source_{uuid.uuid4().hex}.{source_image.filename.rsplit('.', 1)[1]}"
            dest_filename = f"dest_{uuid.uuid4().hex}.{dest_image.filename.rsplit('.', 1)[1]}"
            source_path = os.path.join(temp_dir, source_filename)
            dest_path = os.path.join(temp_dir, dest_filename)

            with open(source_path, "wb") as f:
                f.write(source_content)
            with open(dest_path, "wb") as f:
                f.write(dest_content)

            result_url, face_swap_data = await face_swap(
                source_path, dest_path, task_id=task_id, 
                source_content=source_content, dest_content=dest_content
            )
            
            if background_tasks:
                background_tasks.add_task(cleanup_output_folder)

            return JSONResponse({
                "success": True,
                "data": {
                    "result_image": result_url, 
                    "task_id": task_id,
                    "face_swap_url": face_swap_data.get('face_swap_output_path'),
                    "enhanced_url": face_swap_data.get('enhanced_output_path')
                },
                "error": None
            }, headers={"X-Process-Time": f"{time.time() - start_time:.2f}"})

    except HTTPException as e:
        logger.error(f"Face swap error: {str(e)}")
        progress_tracker[task_id]["status"] = f"Error: {str(e)}"
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "data": None, "error": str(e.detail)},
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        logger.error(f"Face swap error: {str(e)}")
        progress_tracker[task_id]["status"] = f"Error: {str(e)}"
        return JSONResponse(
            status_code=500,
            content={"success": False, "data": None, "error": str(e)},
            headers={"Access-Control-Allow-Origin": "*"}
        )

@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """Get progress of a face swap task with timestamps"""
    try:
        conn = sqlite3.connect("face_swap_data.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT created_at, updated_at FROM face_swap_records WHERE task_id = ?
        """, (task_id,))
        row = cursor.fetchone()
        conn.close()
        created_time = row[0] if row else None
        updated_time = row[1] if row else None
    except Exception as e:
        logger.error(f"Failed to fetch timestamps for task {task_id}: {str(e)}")
        created_time = updated_time = None
    
    progress = progress_tracker.get(task_id, {"status": "Unknown task", "face_swap_url": None, "codeformer_url": None})
    progress["created_at"] = created_time
    progress["updated_at"] = updated_time
    
    return JSONResponse(
        content={
            "task_id": task_id,
            "status": progress["status"],
            "face_swap_url": progress["face_swap_url"],
            "codeformer_url": progress["codeformer_url"],
            "created_at": progress["created_at"],
            "updated_at": progress["updated_at"]
        },
        headers={"Access-Control-Allow-Origin": "*"}
    )

@app.post("/shopify-face-swap")
async def shopify_face_swap(
    user_image: UploadFile = File(...),
    product_image_url: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    start_time = time.time()
    task_id = str(uuid.uuid4())
    progress_tracker[task_id] = {"status": "Starting", "face_swap_url": None, "codeformer_url": None}
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

        progress_tracker[task_id]["status"] = "Downloading product image"
        try:
            headers = {"Authorization": f"Bearer {CONFIG['HF_TOKEN']}"}
            response = requests.get(product_image_url, headers=headers, timeout=10)
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
        elif any(filename in product_image_url for filename in ["DOCTOR.webp", "BOYCHEF1FINAL.webp", "police_investigator.webp", "CULINARY_GIRL.png", "fsoccer.webp", "Pirate_7_1.webp","teacher-girl.webp"]):
            dest_face_idx = 2

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

            result_url, face_swap_data = await face_swap(
                user_path, product_path, dest_face_idx=dest_face_idx, task_id=task_id,
                source_content=user_content, dest_content=product_content
            )
            
            if background_tasks:
                background_tasks.add_task(cleanup_output_folder)

            return JSONResponse({
                "success": True,
                "data": {
                    "result_image": result_url, 
                    "task_id": task_id,
                    "face_swap_url": face_swap_data.get('face_swap_output_path'),
                    "enhanced_url": face_swap_data.get('enhanced_output_path'),
                    "product_image_url": product_image_url
                },
                "error": None
            }, headers={"X-Process-Time": f"{time.time() - start_time:.2f}"})

    except HTTPException as e:
        logger.error(f"Shopify face swap error: {str(e)}")
        progress_tracker[task_id]["status"] = f"Error: {str(e)}"
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
        progress_tracker[task_id]["status"] = f"Error: {str(e)}"
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

# New endpoints for data management
@app.get("/records")
async def records_page(request: Request, limit: int = 50):
    """Render face swap records page"""
    records = get_face_swap_records(limit)
    templates = Jinja2Templates(directory="templates")
    return templates.TemplateResponse(
        "records.html",
        {"request": request, "records": records, "version": app.version}
    )

@app.get("/api/records")
async def get_records(limit: int = 100):
    """Get face swap records from database (JSON API)"""
    records = get_face_swap_records(limit)
    return JSONResponse({
        "success": True,
        "data": records,
        "count": len(records)
    })

@app.get("/records/{task_id}")
async def get_record_by_task_id(task_id: str):
    """Get specific face swap record by task_id"""
    try:
        conn = sqlite3.connect("face_swap_data.db")
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM face_swap_records WHERE task_id = ?
        """, (task_id,))
        
        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            record = dict(zip(columns, row))
            conn.close()
            return JSONResponse({
                "success": True,
                "data": record
            })
        else:
            conn.close()
            raise HTTPException(404, detail="Record not found")
            
    except Exception as e:
        logger.error(f"Failed to get record: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/download-outputs")
def download_outputs():
    """Download all generated output images as a zip"""
    output_dir = CONFIG["OUTPUT_FOLDER"]
    zip_path = "outputs.zip"

    if os.path.exists(zip_path):
        os.remove(zip_path)

    shutil.make_archive("outputs", 'zip', output_dir)

    return FileResponse(
        zip_path,
        media_type='application/zip',
        filename="outputs.zip"
    )

@app.get("/download-data")
def download_data():
    """Download database and data folder as a zip"""
    zip_path = "face_swap_data.zip"
    
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    # Create temporary directory for packaging
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy database
        if os.path.exists("face_swap_data.db"):
            shutil.copy("face_swap_data.db", os.path.join(temp_dir, "face_swap_data.db"))
        
        # Copy data folder
        if os.path.exists(CONFIG["DATA_FOLDER"]):
            shutil.copytree(CONFIG["DATA_FOLDER"], os.path.join(temp_dir, "data"))
        
        # Copy output folder
        if os.path.exists(CONFIG["OUTPUT_FOLDER"]):
            shutil.copytree(CONFIG["OUTPUT_FOLDER"], os.path.join(temp_dir, "output"))
        
        # Create zip
        shutil.make_archive("face_swap_data", 'zip', temp_dir)
    
    return FileResponse(
        zip_path,
        media_type='application/zip',
        filename="face_swap_data.zip"
    )

@app.delete("/cleanup")
async def cleanup_old_files():
    """Manual cleanup of old files"""
    try:
        await cleanup_output_folder()
        
        # Also cleanup old data files
        now = time.time()
        cleanup_count = 0
        
        if os.path.exists(CONFIG["DATA_FOLDER"]):
            for filename in os.listdir(CONFIG["DATA_FOLDER"]):
                file_path = os.path.join(CONFIG["DATA_FOLDER"], filename)
                if os.path.isfile(file_path) and (now - os.path.getmtime(file_path)) > CONFIG["CLEANUP_INTERVAL"]:
                    os.remove(file_path)
                    cleanup_count += 1
                    logger.info(f"Removed old data file: {file_path}")
        
        return JSONResponse({
            "success": True,
            "message": f"Cleanup completed. Removed {cleanup_count} old files.",
            "cleanup_count": cleanup_count
        })
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# Railway deployment
if __name__ == "__main__":
    import uvicorn
    
    # Railway automatically sets PORT
    port = CONFIG["PORT"]
    host = "0.0.0.0"
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
