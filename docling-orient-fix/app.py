"""
Temporary orientation-fix proxy in front of docling-serve.

Workaround for docling-project/docling#3839: docling correctly detects and OCRs
rotated pages internally, but drops the recognized text before final output.
This proxy rotates pages to upright *before* docling ever sees them, sidestepping
the bug entirely. See README.md in this directory for the full story and how to
remove this once upstream is fixed.

Set ORIENT_FIX_ENABLED=false to make this a pure passthrough (no rotation logic)
without removing the service.
"""

import io
import logging
import os

import httpx
import tesserocr
from fastapi import FastAPI, Request, Response
from PIL import Image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("docling-orient-fix")

ENABLED = os.environ.get("ORIENT_FIX_ENABLED", "true").lower() == "true"
UPSTREAM_URL = os.environ.get("DOCLING_UPSTREAM_URL", "http://docling:5001").rstrip("/")

IMAGE_EXTS = {"jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"}

app = FastAPI()

_osd_api = None

# Calibrated against real test pages: a genuine multi-line document (the kind
# actually worth OCR-ing) reads at confidence ~12-14, correctly, both upright and
# rotated. Sparse content (a single short line, or a non-document photo with little/
# no text) reads at <1 or returns no result at all, and is unreliable enough to have
# produced a false-positive rotation in testing. 2.0 sits well clear of both.
MIN_ORIENT_CONFIDENCE = 2.0


def get_osd_api() -> tesserocr.PyTessBaseAPI:
    global _osd_api
    if _osd_api is None:
        _osd_api = tesserocr.PyTessBaseAPI(lang="osd", psm=tesserocr.PSM.OSD_ONLY)
    return _osd_api


def detect_rotation_degrees(pil_image: Image.Image) -> int:
    """Return 0/90/180/270 — the clockwise rotation needed to make the page upright.
    Returns 0 if OSD can't determine orientation reliably (e.g. sparse text, or a
    non-document photo) rather than risk rotating something that didn't need it."""
    api = get_osd_api()
    api.SetImage(pil_image.convert("RGB"))
    osd = api.DetectOrientationScript()
    if not osd:
        return 0
    if osd.get("orient_conf", 0) < MIN_ORIENT_CONFIDENCE:
        log.info(
            "OSD confidence %.2f below threshold (%.1f), leaving orientation unchanged",
            osd.get("orient_conf", 0),
            MIN_ORIENT_CONFIDENCE,
        )
        return 0
    return int(round(osd.get("orient_deg", 0) / 90.0)) * 90 % 360


def fix_image_bytes(data: bytes) -> bytes:
    img = Image.open(io.BytesIO(data))
    img.load()
    fmt = img.format or "PNG"
    deg = detect_rotation_degrees(img)
    if deg == 0:
        return data
    log.info("rotating standalone image by %s degrees", deg)
    rotated = img.rotate(-deg, expand=True)
    buf = io.BytesIO()
    rotated.convert("RGB").save(buf, format=fmt)
    return buf.getvalue()


def fix_pdf_bytes(data: bytes) -> bytes:
    import pypdfium2 as pdfium
    from pypdf import PdfReader, PdfWriter
    import img2pdf

    src_pdf = pdfium.PdfDocument(data)
    reader = PdfReader(io.BytesIO(data))
    writer = PdfWriter()

    any_changed = False
    for i, page in enumerate(reader.pages):
        pdfium_page = src_pdf[i]
        bitmap = pdfium_page.render(scale=2.0)
        pil_img = bitmap.to_pil().convert("RGB")
        deg = detect_rotation_degrees(pil_img)

        if deg == 0:
            # Untouched page keeps its native text layer / vector content.
            writer.add_page(page)
            continue

        any_changed = True
        log.info("rotating PDF page %d by %s degrees (rasterizing this page only)", i, deg)
        rotated = pil_img.rotate(-deg, expand=True)
        buf = io.BytesIO()
        rotated.save(buf, format="PNG")
        page_pdf_bytes = img2pdf.convert(buf.getvalue())
        new_page = PdfReader(io.BytesIO(page_pdf_bytes)).pages[0]
        writer.add_page(new_page)

    if not any_changed:
        return data

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


@app.post("/v1/convert/file")
async def convert_file(request: Request):
    form = await request.form()

    files_out = {}
    data_out: dict[str, list] = {}

    for key, value in form.multi_items():
        if hasattr(value, "filename") and value.filename:
            content = await value.read()
            filename = value.filename
            content_type = value.content_type or "application/octet-stream"

            if ENABLED:
                ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
                try:
                    if ext in IMAGE_EXTS or content_type.startswith("image/"):
                        content = fix_image_bytes(content)
                    elif ext == "pdf" or content_type == "application/pdf":
                        content = fix_pdf_bytes(content)
                except Exception:
                    log.exception(
                        "orientation fix failed for %s, forwarding original file unchanged",
                        filename,
                    )

            files_out[key] = (filename, content, content_type)
        else:
            data_out.setdefault(key, []).append(value)

    data_send = {k: (v[0] if len(v) == 1 else v) for k, v in data_out.items()}

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(f"{UPSTREAM_URL}/v1/convert/file", data=data_send, files=files_out)

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type"),
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def passthrough(path: str, request: Request):
    """Transparent proxy for every other docling-serve endpoint (health checks, async
    convert, etc.) — this service only intercepts /v1/convert/file above."""
    body = await request.body()
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.request(
            request.method,
            f"{UPSTREAM_URL}/{path}",
            params=request.query_params,
            content=body,
            headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
        )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type"),
    )
