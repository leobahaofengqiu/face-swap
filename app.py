import os
import uuid
import hashlib
import tempfile
import time
import logging
import requests
import json
import asyncio
import aiohttp
import aiofiles
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks, HTTPException, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from cachetools import TTLCache
from gradio_client import Client, handle_file
from retry import retry
import shutil
import sqlite3
from dotenv import load_dotenv
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor
import threading

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="High-Quality Face Swap API", version="2.4.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time"],
)

# Configuration for Railway with performance optimizations
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
    "PORT": int(os.getenv("PORT", 8000)),
    "DATABASE_URL": os.getenv("DATABASE_URL", "sqlite:///./face_swap_data.db"),
    # Performance optimizations
    "DOWNLOAD_CHUNK_SIZE": int(os.getenv("DOWNLOAD_CHUNK_SIZE", 65536)),  # 64KB chunks
    "MAX_CONCURRENT_DOWNLOADS": int(os.getenv("MAX_CONCURRENT_DOWNLOADS", 5)),
    "COMPRESSION_LEVEL": int(os.getenv("COMPRESSION_LEVEL", 6)),  # ZIP compression (1-9)
    "STREAM_BUFFER_SIZE": int(os.getenv("STREAM_BUFFER_SIZE", 1024 * 1024)),  # 1MB buffer
}

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=CONFIG["MAX_CONCURRENT_DOWNLOADS"])

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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            file_size INTEGER DEFAULT 0
        )
    """)
    
    # Add indexes for faster queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_task_id ON face_swap_records(task_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON face_swap_records(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON face_swap_records(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_source_hash ON face_swap_records(source_image_hash)")
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

# Initialize database on startup
init_database()

def save_face_swap_record(task_id: str, data: Dict):
    """Save face swap record to database"""
    try:
        conn = sqlite3.connect("face_swap_data.db")
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO face_swap_records 
            (id, task_id, source_image_path, target_image_path, face_swap_output_path, 
             enhanced_output_path, source_image_hash, target_image_hash, dest_face_idx,
             processing_time, status, error_message, metadata, file_size, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
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
            data.get('file_size', 0)
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

def delete_face_swap_record(task_id: str):
    """Delete face swap record and associated files"""
    try:
        conn = sqlite3.connect("face_swap_data.db")
        cursor = conn.cursor()
        
        # Get record first to find file paths
        cursor.execute("SELECT * FROM face_swap_records WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            record = dict(zip(columns, row))
            
            # Delete associated files
            files_to_delete = [
                record.get('source_image_path'),
                record.get('target_image_path'),
                record.get('face_swap_output_path'),
                record.get('enhanced_output_path')
            ]
            
            for file_path in files_to_delete:
                if file_path:
                    # Handle both absolute and relative paths
                    if file_path.startswith('/static/'):
                        actual_path = file_path.replace('/static/', 'static/')
                    elif file_path.startswith('static/'):
                        actual_path = file_path
                    else:
                        actual_path = file_path
                    
                    if os.path.exists(actual_path):
                        try:
                            os.remove(actual_path)
                            logger.info(f"Deleted file: {actual_path}")
                        except Exception as e:
                            logger.error(f"Failed to delete file {actual_path}: {str(e)}")
            
            # Delete database record
            cursor.execute("DELETE FROM face_swap_records WHERE task_id = ?", (task_id,))
            conn.commit()
            conn.close()
            
            return True, record
        else:
            conn.close()
            return False, None
            
    except Exception as e:
        logger.error(f"Failed to delete record: {str(e)}")
        return False, None

# Verify HF_TOKEN exists
if not CONFIG["HF_TOKEN"]:
    logger.error("Hugging Face token (HF_TOKEN) not found in environment variables")
    raise Exception("Hugging Face token (HF_TOKEN) is required")

# Create directories
for folder in [CONFIG["STATIC_DIR"], CONFIG["UPLOAD_FOLDER"], CONFIG["OUTPUT_FOLDER"], CONFIG["DATA_FOLDER"]]:
    os.makedirs(folder, exist_ok=True)

# Optimized StaticFiles with better caching and range support
class OptimizedStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            response = await super().get_response(path, scope)
            response.headers.update({
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Cache-Control": "public, max-age=86400",  # Cache for 24 hours
                "Content-Disposition": "inline",
                "Accept-Ranges": "bytes"  # Enable range requests
            })
            return response
        except Exception as e:
            logger.error(f"Failed to serve static file {path}: {str(e)}")
            raise HTTPException(404, detail=f"Static file {path} not found")

app.mount("/static", OptimizedStaticFiles(directory=CONFIG["STATIC_DIR"]), name="static")

# Cache setup
cache = TTLCache(maxsize=CONFIG["MAX_CACHE_SIZE"], ttl=CONFIG["CACHE_TTL"])
progress_tracker: Dict[str, Dict[str, str]] = {}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in CONFIG["ALLOWED_EXTENSIONS"]

def validate_image(file_path: str) -> bool:
    try:
        with Image.open(file_path) as img:
            img.verify()
            if os.path.getsize(file_path) < CONFIG["MIN_IMAGE_SIZE"]:
                return False
        return True
    except Exception as e:
        logger.error(f"Image validation failed for {file_path}: {str(e)}")
        return False

async def validate_image_url_async(url: str) -> bool:
    """Async version of URL validation for better performance"""
    try:
        headers = {"Authorization": f"Bearer {CONFIG['HF_TOKEN']}"}
        async with aiohttp.ClientSession() as session:
            async with session.head(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status != 200:
                    return False
                content_type = response.headers.get("content-type", "")
                if not content_type.startswith("image/"):
                    return False
                content_length = int(response.headers.get("content-length", 0))
                if content_length < CONFIG["MIN_IMAGE_SIZE"]:
                    return False
                return True
    except Exception as e:
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

# Optimized ZIP creation function
def create_zip_stream(file_paths: List[Tuple[str, str]], compression_level: int = 6):
    """Create ZIP file as streaming response for faster downloads"""
    def generate():
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=compression_level) as zipf:
            for file_path, archive_name in file_paths:
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    zipf.write(file_path, archive_name)
                    # Yield data in chunks to avoid memory issues
                    if buffer.tell() > CONFIG["STREAM_BUFFER_SIZE"]:
                        buffer.seek(0)
                        chunk = buffer.read()
                        buffer = io.BytesIO()
                        yield chunk
        
        # Final chunk
        buffer.seek(0)
        final_chunk = buffer.read()
        if final_chunk:
            yield final_chunk
    
    return generate

async def download_image_async(url: str) -> bytes:
    """Async image download with better performance"""
    try:
        headers = {"Authorization": f"Bearer {CONFIG['HF_TOKEN']}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                return await response.read()
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {str(e)}")
        raise

# Face swap functions (keeping existing implementation but with optimizations)
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
        progress_tracker[task_id]["face_swap_url"] = face_swap_url

        # Enhance with CodeFormer
        progress_tracker[task_id]["status"] = "Enhancing with CodeFormer"
        codeformer_client = Client(
            "sczhou/CodeFormer",
            hf_token=CONFIG["HF_TOKEN"],
            httpx_kwargs={"timeout": CONFIG["GRADIO_TIMEOUT"]}
        )
        
        codeformer_result = codeformer_client.predict(
            image=handle_file(image_path),
            face_align=True,
            background_enhance=True,
            face_upsample=True,
            upscale=1,
            codeformer_fidelity=0.8,
            api_name="/predict"
        )
        
        if not codeformer_result:
            raise ValueError("No valid output from CodeFormer")
        
        # Get original dimensions
        with Image.open(dest_image_path) as dest_img:
            original_width, original_height = dest_img.size

        # Handle CodeFormer result
        if codeformer_result.startswith(('http://', 'https://')):
            content = await download_image_async(codeformer_result)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            with Image.open(temp_file_path) as img:
                img = img.resize((original_width, original_height), Image.Resampling.LANCZOS)
                output_filename = f"enhanced_{task_id}_{uuid.uuid4().hex}.png"
                output_path = os.path.join(CONFIG["OUTPUT_FOLDER"], output_filename)
                img.save(output_path, format="PNG")
            
            os.unlink(temp_file_path)
            codeformer_url = f"/static/output/{output_filename}"
            progress_tracker[task_id]["codeformer_url"] = codeformer_url
            return face_swap_url, codeformer_url
        else:
            if not os.path.exists(codeformer_result) or not validate_image(codeformer_result):
                raise ValueError("Invalid CodeFormer output")
            
            with Image.open(codeformer_result) as img:
                img = img.resize((original_width, original_height), Image.Resampling.LANCZOS)
                output_filename = f"enhanced_{task_id}_{uuid.uuid4().hex}.png"
                output_path = os.path.join(CONFIG["OUTPUT_FOLDER"], output_filename)
                img.save(output_path, format="PNG")
            
            codeformer_url = f"/static/output/{output_filename}"
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
            except Exception as e:
                logger.warning(f"Failed to close CodeFormer client: {str(e)}")

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
        progress_tracker[task_id] = {"status": "Starting", "face_swap_url": None, "codeformer_url": None}
        
        if not all([validate_image(source_image), validate_image(dest_image)]):
            raise ValueError("Invalid input files")

        # Save permanent copies
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
        client = Client(
            "Dentro/face-swap",
            hf_token=CONFIG["HF_TOKEN"],
            httpx_kwargs={"timeout": CONFIG["GRADIO_TIMEOUT"]}
        )

        progress_tracker[task_id]["status"] = f"Processing face swap with destination face {dest_face_idx}"
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_output:
            temp_output_path = temp_output.name
        
        result = client.predict(
            sourceImage=handle_file(source_image),
            sourceFaceIndex=1,
            destinationImage=handle_file(dest_image),
            destinationFaceIndex=dest_face_idx,
            api_name="/predict"
        )

        if result and os.path.exists(result) and validate_image(result):
            shutil.copy(result, temp_output_path)
            
            face_swap_filename = f"face_swap_{task_id}_{uuid.uuid4().hex}.png"
            face_swap_path = os.path.join(CONFIG["OUTPUT_FOLDER"], face_swap_filename)
            shutil.copy(result, face_swap_path)
            face_swap_url = f"/static/output/{face_swap_filename}"
            progress_tracker[task_id]["face_swap_url"] = face_swap_url
            progress_tracker[task_id]["status"] = f"Face swap succeeded"
        else:
            raise ValueError(f"Face swap failed for destination face index {dest_face_idx}")

        progress_tracker[task_id]["status"] = "Applying enhancement pipeline"
        face_swap_url, codeformer_url = await high_quality_enhance(temp_output_path, task_id, dest_image)
        
        processing_time = time.time() - start_time
        
        # Calculate file size
        file_size = 0
        if codeformer_url:
            output_path = codeformer_url.replace('/static/', 'static/')
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
        
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
            'file_size': file_size,
            'metadata': {
                'gradio_result': result,
                'enhancement_applied': True
            }
        }
        
        save_face_swap_record(task_id, face_swap_data)
        
        progress_tracker[task_id]["status"] = f"Completed face swap"
        return codeformer_url, face_swap_data

    except Exception as e:
        processing_time = time.time() - start_time
        progress_tracker[task_id]["status"] = f"Error: {str(e)}"
        
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
            except FileNotFoundError:
                pass
        if client:
            try:
                client.close()
            except Exception as e:
                logger.warning(f"Failed to close Gradio client: {str(e)}")

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

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
async def index(request: Request):
    try:
        templates = Jinja2Templates(directory="templates")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result_image": None, "version": app.version}
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
    
    try:
        if not (source_image.filename and dest_image.filename):
            raise HTTPException(400, detail="No file selected")
        
        if not (allowed_file(source_image.filename) and allowed_file(dest_image.filename)):
            raise HTTPException(400, detail="Invalid file format. Only PNG, JPG, JPEG, WEBP allowed")

        source_content = await source_image.read()
        dest_content = await dest_image.read()
        
        if len(source_content) > CONFIG["MAX_FILE_SIZE"] or len(dest_content) > CONFIG["MAX_FILE_SIZE"]:
            raise HTTPException(400, detail=f"File size exceeds {CONFIG['MAX_FILE_SIZE'] / (1024 * 1024)}MB")

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
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "data": None, "error": str(e.detail)},
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "data": None, "error": str(e)},
            headers={"Access-Control-Allow-Origin": "*"}
        )

@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    progress = progress_tracker.get(task_id, {"status": "Unknown task", "face_swap_url": None, "codeformer_url": None})
    return JSONResponse(
        content={
            "task_id": task_id,
            "status": progress["status"],
            "face_swap_url": progress["face_swap_url"],
            "codeformer_url": progress["codeformer_url"]
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
    
    temp_file_path = None
    try:
        if not user_image.filename:
            raise HTTPException(400, detail="User image is required")
        if not product_image_url or not product_image_url.startswith(('http://', 'https://')):
            raise HTTPException(400, detail="Valid product image URL is required")

        user_content = await user_image.read()
        if len(user_content) > CONFIG["MAX_FILE_SIZE"]:
            raise HTTPException(400, detail=f"User image size exceeds {CONFIG['MAX_FILE_SIZE'] / (1024 * 1024)}MB")
        if not allowed_file(user_image.filename):
            raise HTTPException(400, detail="Invalid user image format. Only PNG, JPG, JPEG, WEBP allowed")

        progress_tracker[task_id] = {"status": "Downloading product image", "face_swap_url": None, "codeformer_url": None}
        
        # Use async download for better performance
        try:
            product_content = await download_image_async(product_image_url)
        except Exception as e:
            raise HTTPException(400, detail=f"Failed to download product image: {str(e)}")

        if len(product_content) > CONFIG["MAX_FILE_SIZE"]:
            raise HTTPException(400, detail=f"Product image size exceeds {CONFIG['MAX_FILE_SIZE'] / (1024 * 1024)}MB")

        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as temp_file:
            temp_file.write(product_content)
            temp_file_path = temp_file.name
        
        if not validate_image(temp_file_path):
            raise HTTPException(400, detail="Invalid product image format")

        # Determine face index based on product URL
        dest_face_idx = 1
        if "TEACHER.webp" in product_image_url:
            dest_face_idx = 3
        elif "REDKNIGHT.webp" in product_image_url:
            dest_face_idx = 4
        elif any(filename in product_image_url for filename in ["DOCTOR.webp", "BOYCHEF1FINAL.webp", "police_investigator.webp", "CULINARY_GIRL.png", "fsoccer.webp", "Pirate_7_1.webp"]):
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
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except FileNotFoundError:
                pass
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "data": None, "error": str(e.detail)},
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except FileNotFoundError:
                pass
        return JSONResponse(
            status_code=500,
            content={"success": False, "data": None, "error": str(e)},
            headers={"Access-Control-Allow-Origin": "*"}
        )

# New endpoints for source images and better data management

@app.get("/records")
async def get_records(limit: int = 100):
    """Get face swap records from database"""
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
        
        cursor.execute("SELECT * FROM face_swap_records WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            record = dict(zip(columns, row))
            conn.close()
            return JSONResponse({"success": True, "data": record})
        else:
            conn.close()
            raise HTTPException(404, detail="Record not found")
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.delete("/records/{task_id}")
async def delete_record(task_id: str):
    """Delete specific face swap record and its files"""
    try:
        deleted, record = delete_face_swap_record(task_id)
        
        if deleted:
            return JSONResponse({
                "success": True,
                "message": f"Record {task_id} and associated files deleted successfully",
                "deleted_record": record
            })
        else:
            raise HTTPException(404, detail="Record not found")
            
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "error": str(e.detail)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# Source image endpoints
@app.get("/download-source/{task_id}")
async def download_source_image(task_id: str):
    """Download source image for specific task"""
    try:
        conn = sqlite3.connect("face_swap_data.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT source_image_path FROM face_swap_records WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0]:
            source_path = row[0]
            if os.path.exists(source_path):
                return FileResponse(
                    source_path,
                    media_type='application/octet-stream',
                    filename=f"source_{task_id}.png",
                    headers={
                        "Access-Control-Allow-Origin": "*",
                        "Cache-Control": "public, max-age=3600"
                    }
                )
            else:
                raise HTTPException(404, detail="Source image file not found")
        else:
            raise HTTPException(404, detail="Source image not found for this task")
            
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "error": str(e.detail)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/download-all-sources")
async def download_all_source_images():
    """Download all source images as a ZIP file"""
    try:
        conn = sqlite3.connect("face_swap_data.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT task_id, source_image_path FROM face_swap_records WHERE source_image_path IS NOT NULL")
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            raise HTTPException(404, detail="No source images found")
        
        # Prepare file paths for ZIP creation
        file_paths = []
        for task_id, source_path in rows:
            if source_path and os.path.exists(source_path):
                # Use task_id in filename for better organization
                archive_name = f"source_{task_id}.png"
                file_paths.append((source_path, archive_name))
        
        if not file_paths:
            raise HTTPException(404, detail="No accessible source image files found")
        
        # Create streaming ZIP response
        generator = create_zip_stream(file_paths, CONFIG["COMPRESSION_LEVEL"])
        
        return StreamingResponse(
            generator(),
            media_type='application/zip',
            headers={
                "Content-Disposition": "attachment; filename=all_source_images.zip",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "error": str(e.detail)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/download-outputs")
async def download_outputs():
    """Download all generated output images as a ZIP with streaming"""
    try:
        output_dir = CONFIG["OUTPUT_FOLDER"]
        
        if not os.path.exists(output_dir):
            raise HTTPException(404, detail="Output directory not found")
        
        # Get all files in output directory
        files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
        
        if not files:
            raise HTTPException(404, detail="No output files found")
        
        # Prepare file paths for ZIP creation
        file_paths = [(os.path.join(output_dir, f), f) for f in files]
        
        # Create streaming ZIP response
        generator = create_zip_stream(file_paths, CONFIG["COMPRESSION_LEVEL"])
        
        return StreamingResponse(
            generator(),
            media_type='application/zip',
            headers={
                "Content-Disposition": "attachment; filename=all_outputs.zip",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "error": str(e.detail)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/download-data")
async def download_data():
    """Download database and data folder as a ZIP with streaming"""
    try:
        file_paths = []
        
        # Add database file
        if os.path.exists("face_swap_data.db"):
            file_paths.append(("face_swap_data.db", "face_swap_data.db"))
        
        # Add data folder files
        if os.path.exists(CONFIG["DATA_FOLDER"]):
            for root, dirs, files in os.walk(CONFIG["DATA_FOLDER"]):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Create relative path for archive
                    archive_path = os.path.relpath(file_path, ".")
                    file_paths.append((file_path, archive_path))
        
        # Add output folder files
        if os.path.exists(CONFIG["OUTPUT_FOLDER"]):
            for root, dirs, files in os.walk(CONFIG["OUTPUT_FOLDER"]):
                for file in files:
                    file_path = os.path.join(root, file)
                    archive_path = os.path.relpath(file_path, ".")
                    file_paths.append((file_path, archive_path))
        
        if not file_paths:
            raise HTTPException(404, detail="No data files found")
        
        # Create streaming ZIP response
        generator = create_zip_stream(file_paths, CONFIG["COMPRESSION_LEVEL"])
        
        return StreamingResponse(
            generator(),
            media_type='application/zip',
            headers={
                "Content-Disposition": "attachment; filename=complete_face_swap_data.zip",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "error": str(e.detail)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/download-image/{task_id}")
async def download_specific_image(task_id: str, image_type: str = "enhanced"):
    """Download specific image by task_id and type (enhanced, face_swap, source, target)"""
    try:
        conn = sqlite3.connect("face_swap_data.db")
        cursor = conn.cursor()
        
        if image_type == "enhanced":
            cursor.execute("SELECT enhanced_output_path FROM face_swap_records WHERE task_id = ?", (task_id,))
            filename_prefix = "enhanced"
        elif image_type == "face_swap":
            cursor.execute("SELECT face_swap_output_path FROM face_swap_records WHERE task_id = ?", (task_id,))
            filename_prefix = "face_swap"
        elif image_type == "source":
            cursor.execute("SELECT source_image_path FROM face_swap_records WHERE task_id = ?", (task_id,))
            filename_prefix = "source"
        elif image_type == "target":
            cursor.execute("SELECT target_image_path FROM face_swap_records WHERE task_id = ?", (task_id,))
            filename_prefix = "target"
        else:
            raise HTTPException(400, detail="Invalid image type. Use: enhanced, face_swap, source, or target")
        
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0]:
            image_path = row[0]
            
            # Handle URL paths (convert to file paths)
            if image_path.startswith('/static/'):
                actual_path = image_path.replace('/static/', 'static/')
            elif image_path.startswith('static/'):
                actual_path = image_path
            else:
                actual_path = image_path
            
            if os.path.exists(actual_path):
                return FileResponse(
                    actual_path,
                    media_type='application/octet-stream',
                    filename=f"{filename_prefix}_{task_id}.png",
                    headers={
                        "Access-Control-Allow-Origin": "*",
                        "Cache-Control": "public, max-age=3600"
                    }
                )
            else:
                raise HTTPException(404, detail=f"{image_type.title()} image file not found")
        else:
            raise HTTPException(404, detail=f"{image_type.title()} image not found for this task")
            
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "error": str(e.detail)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

async def cleanup_output_folder():
    """Cleanup old files"""
    try:
        now = time.time()
        cleanup_count = 0
        
        for folder in [CONFIG["OUTPUT_FOLDER"], CONFIG["DATA_FOLDER"]]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path) and (now - os.path.getmtime(file_path)) > CONFIG["CLEANUP_INTERVAL"]:
                        os.remove(file_path)
                        cleanup_count += 1
                        logger.info(f"Removed old file: {file_path}")
        
        return cleanup_count
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        return 0

@app.delete("/cleanup")
async def cleanup_old_files():
    """Manual cleanup of old files"""
    try:
        cleanup_count = await cleanup_output_folder()
        
        return JSONResponse({
            "success": True,
            "message": f"Cleanup completed. Removed {cleanup_count} old files.",
            "cleanup_count": cleanup_count
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/stats")
async def get_stats():
    """Get statistics about the face swap API"""
    try:
        conn = sqlite3.connect("face_swap_data.db")
        cursor = conn.cursor()
        
        # Get various statistics
        cursor.execute("SELECT COUNT(*) FROM face_swap_records")
        total_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM face_swap_records WHERE status = 'completed'")
        completed_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM face_swap_records WHERE status = 'failed'")
        failed_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(processing_time) FROM face_swap_records WHERE processing_time IS NOT NULL")
        avg_processing_time = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT SUM(file_size) FROM face_swap_records WHERE file_size IS NOT NULL")
        total_file_size = cursor.fetchone()[0] or 0
        
        conn.close()
        
        # Get disk usage
        output_size = 0
        data_size = 0
        
        if os.path.exists(CONFIG["OUTPUT_FOLDER"]):
            for root, dirs, files in os.walk(CONFIG["OUTPUT_FOLDER"]):
                output_size += sum(os.path.getsize(os.path.join(root, file)) for file in files)
        
        if os.path.exists(CONFIG["DATA_FOLDER"]):
            for root, dirs, files in os.walk(CONFIG["DATA_FOLDER"]):
                data_size += sum(os.path.getsize(os.path.join(root, file)) for file in files)
        
        return JSONResponse({
            "success": True,
            "data": {
                "total_records": total_records,
                "completed_records": completed_records,
                "failed_records": failed_records,
                "success_rate": (completed_records / total_records * 100) if total_records > 0 else 0,
                "avg_processing_time": round(avg_processing_time, 2),
                "total_file_size_mb": round(total_file_size / (1024 * 1024), 2),
                "output_folder_size_mb": round(output_size / (1024 * 1024), 2),
                "data_folder_size_mb": round(data_size / (1024 * 1024), 2),
                "total_disk_usage_mb": round((output_size + data_size) / (1024 * 1024), 2)
            }
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# Railway deployment
if __name__ == "__main__":
    import uvicorn
    
    port = CONFIG["PORT"]
    host = "0.0.0.0"
    
    logger.info(f"Starting optimized server on {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
        access_log=True
    )
