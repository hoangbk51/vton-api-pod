import asyncio
import io
import logging
import os
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
from fashn_vton import TryOnPipeline

load_dotenv()

# ---------- Config (đọc từ env) ----------
API_KEY = os.getenv("API_KEY")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")
WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "./weights")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./api_outputs"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)

if not API_KEY:
    raise RuntimeError("Thiếu biến môi trường API_KEY")

ALLOWED_CATEGORIES = {"tops", "bottoms", "one-pieces"}
ALLOWED_PHOTO_TYPES = {"model", "flat-lay"}

RESULT_NAME_RE = re.compile(r"[0-9a-f]{32}\.jpg")
log = logging.getLogger("tryon")
logging.basicConfig(level=logging.INFO)

pipeline: TryOnPipeline | None = None
# Một GPU chỉ nên chạy một inference tại một thời điểm
gpu_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Loading FASHN pipeline...")
    pipeline = TryOnPipeline(weights_dir=WEIGHTS_DIR, device="cuda")
    log.info("Pipeline ready")
    yield


app = FastAPI(title="Virtual Try-On API", lifespan=lifespan)


# ---------- Auth ----------
def require_api_key(x_api_key: str | None = Header(default=None)):
    if x_api_key is None or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---------- Helpers ----------
async def read_limited(upload: UploadFile, field: str) -> bytes:
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{field} vượt quá {MAX_UPLOAD_MB} MB")
    if not data:
        raise HTTPException(400, f"{field} rỗng")
    return data


def decode_image(data: bytes, field: str) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
        return img
    except Exception:
        raise HTTPException(400, f"{field} không phải ảnh hợp lệ")


def validate_params(category: str, garment_photo_type: str, num_timesteps: int):
    if category not in ALLOWED_CATEGORIES:
        raise HTTPException(400, f"category phải thuộc {sorted(ALLOWED_CATEGORIES)}")
    if garment_photo_type not in ALLOWED_PHOTO_TYPES:
        raise HTTPException(400, f"garment_photo_type phải thuộc {sorted(ALLOWED_PHOTO_TYPES)}")
    if not 1 <= num_timesteps <= 50:
        raise HTTPException(400, "num_timesteps phải nằm trong khoảng 1–50")


def run_tryon(person, garment, category, garment_photo_type, num_timesteps, seed) -> str:
    """Chạy đồng bộ trên GPU (gọi qua asyncio.to_thread). Trả về tên file đã lưu."""
    with torch.inference_mode():
        result = pipeline(
            person_image=person,
            garment_image=garment,
            category=category,
            garment_photo_type=garment_photo_type,
            num_samples=1,
            num_timesteps=num_timesteps,
            guidance_scale=1.5,
            seed=seed,
            segmentation_free=True,
        )
    filename = f"{uuid.uuid4().hex}.jpg"
    result.images[0].convert("RGB").save(
        OUTPUT_DIR / filename, format="JPEG", quality=95, subsampling=0
    )
    return filename


async def generate(person_image, garment_image, category, garment_photo_type, num_timesteps, seed):
    validate_params(category, garment_photo_type, num_timesteps)

    person_bytes = await read_limited(person_image, "person_image")
    garment_bytes = await read_limited(garment_image, "garment_image")
    person = decode_image(person_bytes, "person_image")
    garment = decode_image(garment_bytes, "garment_image")

    t0 = time.perf_counter()
    async with gpu_lock:
        filename = await asyncio.to_thread(
            run_tryon, person, garment, category, garment_photo_type, num_timesteps, seed
        )
    total = time.perf_counter() - t0
    log.info("TRYON done in %.3fs -> %s", total, filename)
    return filename, round(total, 3)


# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True, "pipeline_loaded": pipeline is not None}


@app.post("/v1/tryon", dependencies=[Depends(require_api_key)])
async def tryon(
    request: Request,
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form("tops"),
    garment_photo_type: str = Form("model"),
    num_timesteps: int = Form(10),
    seed: int = Form(42),
):
    """Trả JSON chứa URL ảnh kết quả."""
    filename, total = await generate(
        person_image, garment_image, category, garment_photo_type, num_timesteps, seed
    )
    return {
        "ok": True,
        "id": filename[:-4],
        "url": f"{PUBLIC_BASE_URL}/v1/results/{filename}",
        "server_total": total,
    }


@app.post("/v1/tryon/raw", dependencies=[Depends(require_api_key)])
async def tryon_raw(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form("tops"),
    garment_photo_type: str = Form("model"),
    num_timesteps: int = Form(10),
    seed: int = Form(42),
):
    """Trả thẳng file JPEG."""
    filename, _ = await generate(
        person_image, garment_image, category, garment_photo_type, num_timesteps, seed
    )
    return FileResponse(OUTPUT_DIR / filename, media_type="image/jpeg")


@app.get("/v1/results/{filename}")
def get_result(filename: str):
    if not RESULT_NAME_RE.fullmatch(filename):
        raise HTTPException(404, "Not found")
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="image/jpeg")