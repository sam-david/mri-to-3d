"""2D radiographs (X-ray: CR / DX) -> viewer-ready PNGs.

X-rays are single projections, so there is nothing to reconstruct in 3D; the viewer shows them
as an image set next to the 3D models. Pixels are windowed (DICOM VOI LUT / window settings,
else robust percentiles), MONOCHROME1 is inverted so bone is white, and the long side is capped.

Only pixel data is exported, and no DICOM header fields are copied. Some radiographs have
text burned into the pixels (site name, sometimes patient details). Check the images before
sharing them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

XRAY_MODALITIES = {"CR", "DX", "RF", "MG", "XA", "PX", "IO"}
MAX_SIDE = 2048
SKIP_SUFFIXES = {".exe", ".dll", ".inf", ".htm", ".html", ".xml", ".txt", ".chm", ".ini", ".png",
                 ".jpg", ".jpeg", ".pdf", ".js", ".css", ".vbs", ".cab", ".json", ".md"}


def _read(path: Path):
    import pydicom
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception:
        return None
    if ds.get("Modality") not in XRAY_MODALITIES or not hasattr(ds, "Rows"):
        return None
    return ds


def _display(ds) -> np.ndarray:
    """Pixel data -> uint8 for display."""
    import pydicom
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
    full = pydicom.dcmread(ds.filename)
    a = full.pixel_array
    if a.ndim == 3 and a.shape[-1] in (3, 4):  # already colour/RGB (rare): just scale
        return np.clip(a[..., :3], 0, 255).astype(np.uint8)
    a = apply_modality_lut(a, full).astype(np.float32)
    if "VOILUTSequence" in full or "WindowCenter" in full:
        a = apply_voi_lut(a, full, prefer_lut=True).astype(np.float32)
        lo, hi = float(a.min()), float(a.max())
    else:  # no window in the header: robust range over all pixels (works for either polarity)
        lo, hi = np.percentile(a, [0.5, 99.7])
        if hi <= lo:
            lo, hi = float(a.min()), float(a.max())
    out = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    if full.get("PhotometricInterpretation") == "MONOCHROME1":
        out = 1 - out
    return (out * 255).astype(np.uint8)


def convert(dicom_dir: Path, out_dir: Path, info: dict) -> dict:
    """Find radiographs under dicom_dir and write OUT/images/*.png + OUT/manifest.json."""
    from PIL import Image

    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    found = []
    for f in sorted(Path(dicom_dir).rglob("*")):
        if f.is_file() and f.suffix.lower() not in SKIP_SUFFIXES and (ds := _read(f)):
            found.append(ds)
    if not found:
        raise SystemExit(f"no radiographs (CR/DX) found under {dicom_dir}")
    images, groups = [], {}
    for ds in sorted(found, key=lambda d: (str(d.get("StudyDate", "")), str(d.get("StudyDescription", "")),
                                           int(d.get("SeriesNumber", 0) or 0), int(d.get("InstanceNumber", 0) or 0))):
        study = str(ds.get("StudyDescription") or ds.get("BodyPartExamined") or "X-ray").strip()
        raw = str(ds.get("ViewPosition") or ds.get("SeriesDescription") or "").strip()
        view = raw.upper() if len(raw) <= 3 else raw.title() or "View"  # AP, PA, LL stay upper-case
        date = str(ds.get("StudyDate") or "")
        n = groups[study] = groups.get(study, 0) + 1
        img = Image.fromarray(_display(ds))
        if max(img.size) > MAX_SIDE:
            img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        name = f"{len(images) + 1:02d}.png"
        img.save(out_dir / "images" / name, optimize=True)
        spacing = ds.get("ImagerPixelSpacing") or ds.get("PixelSpacing")
        images.append({"src": f"images/{name}", "study": study, "view": view, "index": n,
                       "date": f"{date[:4]}-{date[4:6]}-{date[6:]}" if len(date) == 8 else date,
                       "width": img.size[0], "height": img.size[1],
                       "pixel_mm": round(float(spacing[0]) * ds.Columns / img.size[0], 4) if spacing else None})
    manifest = {"kind": "images", **info, "images": images}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
