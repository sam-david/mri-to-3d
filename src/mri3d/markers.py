"""Anatomical markers placed from the reconstructed geometry.

Clinical MRI rarely resolves a nerve root or a neural foramen (a few mm across) well enough to
segment it. Where a report names a level and side ("left L4-5 foraminal stenosis"),
we can still mark *where* that is, from the patient's own vertebrae, disc and spinal cord:

  foramen  the exit path of the spinal nerve at a disc level. In the cervical spine the root
           exiting at C6-C7 is C7; elsewhere the root takes the number of the vertebra above.

Markers are positions, not measurements: they show location, not the degree of narrowing.
All coordinates are RAS millimetres (scanner space, same as the STLs).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

FORAMEN_ANGLE_DEG = {"C": 45.0, "T": 0.0, "L": 0.0}  # axial-plane angle from pure lateral, toward anterior
FORAMEN_DROOP_DEG = {"C": 10.0, "T": 0.0, "L": 15.0}  # how steeply the root runs downward


def exiting_root(upper: str, lower: str) -> str:
    """Which nerve root exits between two vertebrae."""
    region = lower[0]
    if region == "C":                      # C-roots exit above their own vertebra (C8 below C7)
        return "C8" if upper == "C7" else lower
    if upper == "C7":
        return "C8"
    return upper                           # T/L/S roots exit below their own vertebra


def _unit(v):
    return v / np.linalg.norm(v)


def foramen(stl_dir: Path, level: str, side: str) -> dict:
    """Marker path for the neural foramen at `level` (e.g. "L4-L5") on `side` ("left"/"right")."""
    upper, lower = level.split("-")
    load = lambda n: trimesh.load(stl_dir / f"{n}.stl")  # noqa: E731
    try:
        vu, vl, cord = load(f"vertebrae_{upper}"), load(f"vertebrae_{lower}"), load("spinal_cord")
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"cannot mark {level}: needs vertebrae {upper}, {lower} and the spinal cord ({e})")
    disc_path = stl_dir / f"disc_{level}.stl"
    level_pt = trimesh.load(disc_path).centroid if disc_path.exists() else (vu.centroid + vl.centroid) / 2

    up = _unit(vu.centroid - vl.centroid)                      # local spine axis
    lat0 = np.array([1.0, 0, 0]) * (1 if side == "right" else -1)  # RAS: +x = subject's right
    lat = _unit(lat0 - up * (lat0 @ up))
    ant = _unit(np.cross(up, lat) if side == "right" else np.cross(lat, up))
    if ant[1] < 0:
        ant = -ant

    # Cord cross-section at the disc level: its centre and right/left edge.
    v = cord.vertices
    slab = v[np.abs((v - level_pt) @ up) < 2.5]
    if len(slab) < 10:
        slab = v[np.argsort(np.abs((v - level_pt) @ up))[:200]]
    centre = slab.mean(0)
    centre = centre - up * ((centre - level_pt) @ up)          # snap to the disc plane
    half_width = float(((slab - centre) @ lat).max())

    region = upper[0] if upper[0] in FORAMEN_ANGLE_DEG else "L"
    a, d = np.radians(FORAMEN_ANGLE_DEG[region]), np.radians(FORAMEN_DROOP_DEG[region])
    direction = _unit(lat * np.cos(a) + ant * np.sin(a) - up * np.tan(d))
    start = centre + lat * (half_width + 0.5)
    steps = np.array([0.0, 5.0, 10.0, 15.0, 20.0])

    # Fit the height to this patient's bony gap: slide the path along the spine axis and keep
    # the offset where it sits most evenly between the upper and lower vertebra.
    tu, tl = cKDTree(vu.sample(30000)), cKDTree(vl.sample(30000))
    best, best_h = -np.inf, 0.0
    for h in np.arange(-8.0, 12.01, 0.5):
        pts = start + up * h + direction * steps[:, None]
        du, dl = tu.query(pts[1:4])[0], tl.query(pts[1:4])[0]
        score = np.minimum(du, dl).mean() - 0.3 * np.abs(du - dl).mean()
        if score > best:
            best, best_h = score, h
    start = start + up * best_h
    path = [start + direction * t for t in steps]
    root = exiting_root(upper, lower)
    return {
        "kind": "foramen", "id": f"foramen_{level}_{side}",
        "label": f"{side.capitalize()} {upper}–{lower} foramen ({root} nerve root)",
        "level": level, "side": side, "root": root,
        "path": [np.round(p, 2).tolist() for p in path], "radius": 2.2,
        "fit": {"height_offset_mm": float(best_h), "clearance_mm": round(float(best), 1)},
        "related": [f"vertebrae_{upper}", f"vertebrae_{lower}", f"disc_{level}", "spinal_cord"],
    }


def add(out_dir: Path, spec: str, note: str | None = None) -> dict:
    """spec: 'L4-L5:left'. Adds/replaces the marker in OUT/manifest.json."""
    m = re.fullmatch(r"([CTLS]\d+-[CTLS]\d+):(left|right)", spec.strip(), re.I)
    if not m:
        raise SystemExit(f"bad marker {spec!r}: use LEVEL:SIDE, e.g. L4-L5:left")
    level = "-".join(p.upper() for p in m[1].split("-"))
    mk = foramen(Path(out_dir) / "stl", level, m[2].lower())
    if note:
        mk["note"] = note
    mpath = Path(out_dir) / "manifest.json"
    manifest = json.loads(mpath.read_text())
    manifest["annotations"] = [a for a in manifest.get("annotations", []) if a["id"] != mk["id"]] + [mk]
    mpath.write_text(json.dumps(manifest, indent=1))
    return mk
