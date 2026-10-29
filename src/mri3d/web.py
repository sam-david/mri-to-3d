"""Package built studies into the web viewer.

Two layouts:
  single file  viewer.html with every model inlined (open by double-click, email, AirDrop)
  folder       index.html + models/<study>.glb, for static hosting; each model stays a
               separate file so hosts with per-file size limits work
"""
from __future__ import annotations

import base64
import io
import json
from importlib import resources
from pathlib import Path

import numpy as np
import trimesh

# RAS (z up) -> glTF (y up, +z toward a viewer facing the subject): x=L, y=S, z=A. det = +1.
RAS_TO_GLTF = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)


def _faces_for(volume_cm3: float) -> int:
    return int(np.clip(volume_cm3 * 250, 2500, 12000))


def _to_gltf(points, center):
    p = np.asarray(points, float) - center
    return np.round(p @ RAS_TO_GLTF[:3, :3].T, 2).tolist()


def study_center(study_dir: Path) -> np.ndarray:
    return np.mean([trimesh.load(p).centroid for p in sorted((study_dir / "stl").glob("*.stl"))], axis=0)


def study_glb(study_dir: Path, manifest: dict) -> bytes:
    stl = {p.stem: trimesh.load(p) for p in sorted((study_dir / "stl").glob("*.stl"))}
    center = np.mean([m.centroid for m in stl.values()], axis=0)
    scene = trimesh.Scene()
    for name, m in stl.items():
        target = _faces_for(manifest["structures"].get(name, {}).get("volume_cm3", 10))
        if len(m.faces) > target:
            m = m.simplify_quadric_decimation(face_count=target)
        m.apply_translation(-center)
        m.apply_transform(RAS_TO_GLTF)
        scene.add_geometry(m, node_name=name, geom_name=name)
    buf = io.BytesIO()
    scene.export(buf, file_type="glb")
    return buf.getvalue()


def _image_entry(d: Path, manifest: dict, dest: Path, folder: bool) -> dict:
    """X-ray sets: copy PNGs next to the page (folder) or inline them as data URIs."""
    images = []
    for im in manifest["images"]:
        png = (d / im["src"]).read_bytes()
        if folder:
            rel = f"models/{manifest['id']}/{Path(im['src']).name}"
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            (dest / rel).write_bytes(png)
            src = rel
        else:
            src = "data:image/png;base64," + base64.b64encode(png).decode()
        images.append({**im, "src": src})
    return {k: manifest.get(k) for k in ("kind", "id", "title", "date", "notes")} | {"images": images}


def export(study_dirs: list[Path], dest: Path, folder: bool = False, title: str = "3D Scan Models") -> Path:
    template = resources.files("mri3d").joinpath("viewer.html").read_text()
    studies = []
    if folder:
        (dest / "models").mkdir(parents=True, exist_ok=True)
    for d in study_dirs:
        d = Path(d)
        manifest = json.loads((d / "manifest.json").read_text())
        if manifest.get("kind") == "images":
            studies.append(_image_entry(d, manifest, dest, folder))
            print(f"  {manifest['id']}: {len(manifest['images'])} images")
            continue
        glb = study_glb(d, manifest)
        center = study_center(d)
        entry = {k: manifest.get(k) for k in ("id", "title", "date", "sources", "notes", "highlight", "alignment", "structures")}
        entry["annotations"] = [{**an, "path": _to_gltf(an["path"], center)} for an in manifest.get("annotations", [])]
        if folder:
            (dest / "models" / f"{manifest['id']}.glb").write_bytes(glb)
            entry["src"] = f"models/{manifest['id']}.glb"
        else:
            entry["glb"] = base64.b64encode(glb).decode()
        studies.append(entry)
        print(f"  {manifest['id']}: {len(glb) / 1e6:.1f} MB, {len(manifest['structures'])} structures")
    html = (template.replace("/*__STUDIES__*/[]", json.dumps(studies))
                    .replace("__TITLE__", title))
    out = dest / "index.html" if folder else dest
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out
