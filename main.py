from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from storage3 import SyncStorageClient
from storage3.exceptions import StorageApiError

load_dotenv()


IMAGE_MIME_PREFIX = "image/"
VIDEO_MIME_PREFIX = "video/"
PDF_MIME_TYPE = "application/pdf"


class UploadResponseModel(BaseModel):
    bucket: str = Field(description="Supabase bucket where the file was stored")
    path: str = Field(description="Relative object path in the bucket")
    full_path: str = Field(description="Storage full path/key returned by Supabase")
    url: str = Field(description="Public or signed URL for the uploaded file")
    url_type: str = Field(description="public or signed")


class ErrorResponseModel(BaseModel):
    detail: str


def _normalize_storage_url(raw_url: str) -> str:
    url = raw_url.rstrip("/")
    if url.endswith("/storage/v1"):
        return url
    return f"{url}/storage/v1"


def _get_required_env() -> tuple[str, str]:
    raw_url = os.getenv("SUPABASE_URL") or os.getenv("SUPABASE_TEST_URL")
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_TEST_KEY")
    )

    if not raw_url:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_TEST_URL in .env")
    if not key:
        raise RuntimeError(
            "Missing SUPABASE_SERVICE_ROLE_KEY, SUPABASE_KEY, or SUPABASE_TEST_KEY in .env"
        )

    return _normalize_storage_url(raw_url), key


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        url, key = _get_required_env()
    except RuntimeError as exc:
        raise RuntimeError(f"Invalid environment configuration: {exc}") from exc

    headers = {"apiKey": key, "Authorization": f"Bearer {key}"}
    app.state.storage = SyncStorageClient(url, headers)
    app.state.default_bucket = os.getenv("SUPABASE_BUCKET", "uploads")
    app.state.default_bucket_public = _get_bool_env("SUPABASE_BUCKET_PUBLIC", True)
    try:
        yield
    finally:
        app.state.storage.session.close()


app = FastAPI(
    title="Supabase Storage Upload API",
    description=(
        "API para subir archivos a Supabase Storage. "
        "Prueba los endpoints directamente desde Swagger en /docs."
    ),
    version="1.1.0",
    lifespan=lifespan,
    contact={"name": "Storage API"},
    openapi_tags=[
        {"name": "Health", "description": "Estado del servicio"},
        {
            "name": "Upload",
            "description": "Endpoints de carga para imagen, video y PDF",
        },
    ],
)


def _ensure_content_type(file: UploadFile, expected: str) -> None:
    content_type = file.content_type or ""
    if expected == "image" and not content_type.startswith(IMAGE_MIME_PREFIX):
        raise HTTPException(
            status_code=400, detail="Solo se permiten archivos de imagen"
        )
    if expected == "video" and not content_type.startswith(VIDEO_MIME_PREFIX):
        raise HTTPException(
            status_code=400, detail="Solo se permiten archivos de video"
        )
    if expected == "pdf" and content_type != PDF_MIME_TYPE:
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")


def _is_bucket_not_found_error(exc: StorageApiError) -> bool:
    return "bucket not found" in exc.message.lower() or exc.status == 404


async def _upload_to_supabase(
    *,
    file: UploadFile,
    folder: str,
) -> UploadResponseModel:
    if not file.filename:
        raise HTTPException(status_code=400, detail="The uploaded file has no filename")

    target_bucket = app.state.default_bucket
    suffix = Path(file.filename).suffix
    generated_name = f"{uuid4().hex}{suffix}"

    clean_folder = folder.strip("/")
    object_path = f"{clean_folder}/{generated_name}" if clean_folder else generated_name

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")

    content_type = file.content_type or "application/octet-stream"

    try:
        _ensure_bucket_exists(target_bucket)
        bucket_client = app.state.storage.from_(target_bucket)
        try:
            result = bucket_client.upload(
                object_path,
                content,
                {"content-type": content_type, "upsert": "true"},
            )
        except StorageApiError as upload_exc:
            # Some projects return "Bucket not found" at upload time even after a check.
            if not _is_bucket_not_found_error(upload_exc):
                raise
            _create_bucket_if_missing(target_bucket)
            result = bucket_client.upload(
                object_path,
                content,
                {"content-type": content_type, "upsert": "true"},
            )

        url = bucket_client.get_public_url(object_path)
        url_type = "public"

    except StorageApiError as exc:
        detail = f"Upload failed: {exc.message}"
        if exc.status in {401, 403}:
            detail = (
                "Upload failed due to permissions. Use SUPABASE_SERVICE_ROLE_KEY "
                "to allow automatic bucket creation and uploads from backend."
            )
        raise HTTPException(status_code=500, detail=detail) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    return UploadResponseModel(
        bucket=target_bucket,
        path=result.path,
        full_path=result.full_path,
        url=url,
        url_type=url_type,
    )


def _ensure_bucket_exists(bucket_name: str) -> None:
    """Create default bucket on demand if it does not exist."""
    try:
        app.state.storage.get_bucket(bucket_name)
        return
    except StorageApiError as exc:
        if not _is_bucket_not_found_error(exc) and exc.status != 400:
            raise
    _create_bucket_if_missing(bucket_name)


def _create_bucket_if_missing(bucket_name: str) -> None:
    try:
        app.state.storage.create_bucket(
            id=bucket_name,
            options={"public": app.state.default_bucket_public},
        )
    except StorageApiError as exc:
        message = exc.message.lower()
        if "already exists" in message or exc.status == 409:
            return
        raise


@app.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/upload",
    tags=["Upload"],
    summary="Carga cualquier archivo",
    response_model=UploadResponseModel,
    responses={400: {"model": ErrorResponseModel}, 500: {"model": ErrorResponseModel}},
)
async def upload_file(
    file: UploadFile = File(..., description="Archivo a subir"),
) -> UploadResponseModel:
    return await _upload_to_supabase(
        file=file,
        folder="uploads",
    )


@app.post(
    "/upload/image",
    tags=["Upload"],
    summary="Carga una imagen",
    response_model=UploadResponseModel,
    responses={400: {"model": ErrorResponseModel}, 500: {"model": ErrorResponseModel}},
)
async def upload_image(
    file: UploadFile = File(..., description="Imagen (image/*)"),
) -> UploadResponseModel:
    _ensure_content_type(file, "image")
    return await _upload_to_supabase(
        file=file,
        folder="images",
    )


@app.post(
    "/upload/video",
    tags=["Upload"],
    summary="Carga un video",
    response_model=UploadResponseModel,
    responses={400: {"model": ErrorResponseModel}, 500: {"model": ErrorResponseModel}},
)
async def upload_video(
    file: UploadFile = File(..., description="Video (video/*)"),
) -> UploadResponseModel:
    _ensure_content_type(file, "video")
    return await _upload_to_supabase(
        file=file,
        folder="videos",
    )


@app.post(
    "/upload/pdf",
    tags=["Upload"],
    summary="Carga un PDF",
    response_model=UploadResponseModel,
    responses={400: {"model": ErrorResponseModel}, 500: {"model": ErrorResponseModel}},
)
async def upload_pdf(
    file: UploadFile = File(..., description="Documento PDF (application/pdf)"),
) -> UploadResponseModel:
    _ensure_content_type(file, "pdf")
    return await _upload_to_supabase(
        file=file,
        folder="documents",
    )
