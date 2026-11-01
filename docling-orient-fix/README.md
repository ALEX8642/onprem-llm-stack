# docling-orient-fix — TEMPORARY workaround, remove when upstream is fixed

**This service exists only because of an unresolved upstream bug:**
[docling-project/docling#3839](https://github.com/docling-project/docling/issues/3839).

## What's actually wrong

Docling correctly detects when a scanned page/image is rotated (90°/180°/270°) and
correctly OCRs the text after internally rotating it — confirmed by tracing docling's
own code: OCR finds the right words at 90%+ confidence. But the recognized text is
then dropped somewhere in docling's document-assembly stage before it reaches the
final output. The end result: upside-down or sideways scans come back with **zero**
extracted text, even though OCR read them correctly internally. `force_full_page_ocr`
does not fix it. This is not a config problem on our side — see the linked issue for
the full trace.

**Precise mechanism** (per the docling maintainer bot's follow-up on the issue): OCR and
layout analysis run in different coordinate spaces for a rotated page. OCR rotates the
image, recognizes text, then transforms cell coordinates back to the original frame —
correctly. Layout analysis runs separately, on the *original, still-rotated* image, so its
clusters stay in the un-rotated coordinate space. Assigning OCR text to a layout cluster
requires spatial overlap between the two — since one side is coordinate-flipped and the
other isn't, nothing overlaps, and every cluster ends up with no text.

## What this service does

Sits between OpenWebUI and the real `docling` service. For every file OpenWebUI sends
for extraction, it:
1. Runs Tesseract OSD (orientation & script detection) to check if the page is rotated.
   Skips rotation if OSD's confidence is below `MIN_ORIENT_CONFIDENCE` (2.0, in `app.py`)
   — calibrated against real test pages: genuine multi-line document content reads at
   confidence ~12-14, correctly; sparse content (a single short line, or a non-document
   photo with little/no text) reads at <1 or returns no result, and produced a real
   false-positive rotation in testing before this gate was added.
2. If it is, physically rotates the image (or just the affected page, for PDFs — pages
   that don't need rotation keep their original text layer untouched) to upright.
3. Forwards the corrected file to the real `docling` service and returns its response
   unchanged.

This avoids the bug entirely by never handing docling a rotated page in the first
place, rather than depending on docling's own (broken) internal correction.

All other requests (health checks, `/v1/convert/file/async`, etc.) pass straight
through to `docling` untouched — this only intercepts `/v1/convert/file`, which is the
one endpoint OpenWebUI's docling loader actually calls.

## How to turn it off without removing anything

Set `ORIENT_FIX_ENABLED=false` on this service (in `docker-compose.yml`) and restart
it. It becomes a pure passthrough proxy — no rotation logic runs, every request goes
to `docling` unchanged. Useful for A/B-testing whether a given extraction problem is
actually caused by this service vs. something else.

## How to remove this entirely once docling-project/docling#3839 is fixed

1. Check the issue — once it's closed with a real fix (not just closed-stale) and a
   new `docling-serve` image has shipped it, pull the updated image and confirm the
   repro in the issue actually comes back with text now.
2. In `docker-compose.yml`: delete the `docling-orient-fix` service block.
3. In OpenWebUI's config, set `rag.docling_server_url` back to `http://docling:5001`
   (it currently points at `http://docling-orient-fix:5001`).
4. Restart `open-webui`.
5. Delete this directory (`~/ai-stack/docling-orient-fix/`).

That's the whole removal — nothing else in the stack depends on this service, and
`docling` itself was never modified.
