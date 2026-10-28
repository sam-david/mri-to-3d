"""Fuse per-series segmentations into one set of smooth 3D meshes.

Clinical MRI is mostly thick 2D slices (3-7 mm). A mesh from one series looks like a stack of
plates along the slice direction. Exams usually contain the same anatomy in two or three planes,
so each structure's mask from every series is softened, resampled into a shared isotropic
world-space grid and averaged, with weights feathered toward each field-of-view edge. Each plane
contributes the directions it resolves well.

Separate studies (e.g. a thoracic and a lumbar exam, or two exam dates) can sit in different
scanner frames. Groups are aligned by translating onto the structures they both contain fully,
preferring rigid bone.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nb
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

from .anatomy import category, level, pretty

FEATHER_MM = 6.0
MIN_STRUCTURE_CM3 = 0.3
MAX_GRID_VOXELS = 20e6
MESH_FACE_CAP = 80000


@dataclass
class Mask:
    data: np.ndarray  # bool
    affine: np.ndarray
    zooms: tuple
    series: str

    @property
    def cm3(self) -> float:
        return float(self.data.sum() * np.prod(self.zooms) / 1000)


def load_group(series_dirs: dict[str, list[Path]]) -> dict[str, list[Mask]]:
    """{series id: [task output dirs]} -> {structure: [Mask per series]}.
    If two tasks label the same structure in one series, the first task listed wins."""
    found: dict[str, dict[str, Mask]] = {}
    for sid, dirs in series_dirs.items():
        for d in dirs:
            for f in sorted(Path(d).glob("*.nii.gz")):
                name = f.name[:-7]
                if sid in found.get(name, {}):
                    continue
                img = nb.load(f)
                arr = np.asarray(img.dataobj) > 0
                if not arr.any():
                    continue
                found.setdefault(name, {})[sid] = Mask(arr, img.affine, img.header.get_zooms()[:3], sid)
    if any(n.startswith("vertebrae_") for n in found):
        found.pop("vertebrae", None)  # combined label is redundant when levels exist
    return {n: list(m.values()) for n, m in found.items()}


def _centroids(group):
    out = {}
    for name, masks in group.items():
        best = max(masks, key=lambda m: m.cm3)
        out[name] = (nb.affines.apply_affine(best.affine, np.argwhere(best.data).mean(0)), best.cm3)
    return out


def align_groups(groups: list[dict[str, list[Mask]]]) -> tuple[dict[str, list[Mask]], list[str]]:
    """Merge groups into the frame of the largest one, translating each by the median centroid
    offset of structures both see near-completely (bones preferred). Returns merged + log."""
    order = sorted(range(len(groups)), key=lambda i: -len(groups[i]))
    merged = {k: list(v) for k, v in groups[order[0]].items()}
    pending, log = order[1:], []
    while pending:
        progress = False
        for i in list(pending):
            a, b = _centroids(merged), _centroids(groups[i])
            shared = [k for k in a if k in b and min(a[k][1], b[k][1]) > 0.7 * max(a[k][1], b[k][1])
                      and category(k) not in ("lung", "fat", "muscle")]
            bones = [k for k in shared if category(k) == "bone"]
            anchors = bones if len(bones) >= 2 else shared
            if not anchors:
                continue
            shifts = np.array([a[k][0] - b[k][0] for k in anchors])
            t = np.median(shifts, axis=0)
            resid = float(np.abs(shifts - t).max())
            T = np.eye(4); T[:3, 3] = t
            for name, masks in groups[i].items():
                merged.setdefault(name, []).extend(
                    Mask(m.data, T @ m.affine, m.zooms, m.series) for m in masks)
            log.append(f"aligned group {i} via {', '.join(pretty(k) for k in anchors[:6])}"
                       f" (shift {np.round(t, 1).tolist()} mm, residual {resid:.1f} mm)")
            pending.remove(i); progress = True
        if not progress:
            for i in pending:
                log.append(f"group {i} skipped: shares no complete structure with the others")
            break
    return merged, log


def _grid(masks):
    pts = []
    for m in masks:
        idx = np.argwhere(m.data)
        lo, hi = idx.min(0) - 2, idx.max(0) + 2
        c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        pts.append(nb.affines.apply_affine(m.affine, c))
    pts = np.vstack(pts)
    lo, hi = pts.min(0), pts.max(0)
    finest = min(min(m.zooms) for m in masks)
    voxel = max(0.5, finest * 0.75, float(np.cbrt(np.prod(hi - lo) / MAX_GRID_VOXELS)))
    return lo, hi, voxel


def fuse(masks: list[Mask]):
    """Weighted average of soft masks on a shared grid -> (prob, grid affine, voxel, n_series)."""
    lo, hi, voxel = _grid(masks)
    shape = np.ceil((hi - lo) / voxel).astype(int) + 1
    aff = np.diag([voxel, voxel, voxel, 1.0]); aff[:3, 3] = lo
    world = nb.affines.apply_affine(aff, np.indices(shape).reshape(3, -1).T)
    acc = np.zeros(len(world), np.float32); wsum = np.zeros_like(acc); nsrc = np.zeros(len(world), np.uint8)
    for m in masks:
        z = np.array(m.zooms, float)
        # Light blur removes the stair-step; across thick slices blur ~half a slice.
        sigma = [max(0.6 * z.min() / zi, 0.5 if zi > 2 * z.min() else 0) for zi in z]
        soft = ndimage.gaussian_filter(m.data.astype(np.float32), sigma=sigma)
        vox = nb.affines.apply_affine(np.linalg.inv(m.affine), world).T
        edge = np.minimum(vox + 0.5, np.array(m.data.shape)[:, None] - 0.5 - vox) * z[:, None]
        w = np.clip(edge / FEATHER_MM, 0, 1).prod(axis=0).astype(np.float32)
        acc += w * ndimage.map_coordinates(soft, vox, order=1, mode="constant", cval=0.0)
        wsum += w; nsrc += w > 0.5
    prob = np.divide(acc, wsum, out=np.zeros_like(acc), where=wsum > 1e-3)
    return prob.reshape(shape), aff, voxel, nsrc.reshape(shape)


def mesh(binary, aff, voxel) -> trimesh.Trimesh | None:
    min_mm3 = 30
    if binary.sum() * voxel ** 3 < min_mm3:
        return None
    v, f, _, _ = measure.marching_cubes(np.pad(binary, 1).astype(np.float32), 0.5)
    m = trimesh.Trimesh(nb.affines.apply_affine(aff, v - 1), f, process=True)
    parts = [p for p in m.split(only_watertight=False) if abs(p.volume) >= min_mm3]
    if not parts:
        return None
    m = trimesh.util.concatenate(parts)
    trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=25)
    if len(m.faces) > MESH_FACE_CAP:
        m = m.simplify_quadric_decimation(face_count=MESH_FACE_CAP)
    m.fix_normals()
    return m


def split_discs(binary, aff, voxel, vert_centroids):
    """Name each connected disc by the vertebrae above and below it."""
    lab, n = ndimage.label(binary)
    order = sorted(vert_centroids.items(), key=lambda kv: -kv[1][2])  # head to tail
    out = {}
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() * voxel ** 3 < 100:
            continue
        z = nb.affines.apply_affine(aff, np.argwhere(comp).mean(0))[2]
        name = f"disc_{i}"
        for (a, ca), (b, cb) in zip(order, order[1:]):
            if cb[2] <= z <= ca[2]:
                name = f"disc_{a.split('_')[1]}-{b.split('_')[1]}"
                break
        out[name] = comp
    return out


def build(groups: list[dict[str, list[Path]]], out_dir: Path, info: dict) -> dict:
    """groups: one {series id: [seg dirs]} per scanner frame. Writes STLs + manifest.json."""
    out_dir = Path(out_dir)
    (out_dir / "stl").mkdir(parents=True, exist_ok=True)
    merged, log = align_groups([load_group(g) for g in groups])
    for line in log:
        print("  " + line)
    n_series = len({sid for g in groups for sid in g})
    meshes, stats, vert = {}, {}, {}
    discs = None
    for name in sorted(merged, key=level):
        prob, aff, voxel, nsrc = fuse(merged[name])
        binary = prob >= 0.5
        cm3 = float(binary.sum() * voxel ** 3 / 1000)
        if name == "intervertebral_discs":
            discs = (binary, aff, voxel)
            continue
        if cm3 < MIN_STRUCTURE_CM3:
            continue
        m = mesh(binary, aff, voxel)
        if m is None:
            continue
        meshes[name] = m
        stats[name] = {"volume_cm3": round(cm3, 2),
                       "multi_series_fraction": round(float((nsrc[binary] >= 2).mean()), 2) if n_series > 1 else None}
        if name.startswith("vertebrae_"):
            vert[name] = m.centroid
        elif name == "sacrum":  # lets the lowest disc be named L5-S1
            vert["vertebrae_S1"] = np.array([*m.centroid[:2], m.bounds[1][2] - 15])
    if discs is not None:
        for dname, comp in split_discs(*discs, vert).items():
            m = mesh(comp, discs[1], discs[2])
            if m is not None:
                meshes[dname] = m
                stats[dname] = {"volume_cm3": round(float(comp.sum() * discs[2] ** 3 / 1000), 2),
                                "multi_series_fraction": None}
    for old in (out_dir / "stl").glob("*.stl"):
        old.unlink()
    for name, m in meshes.items():
        m.export(out_dir / "stl" / f"{name}.stl")
        stats[name].update(category=category(name), label=pretty(name))
    manifest = {**info, "alignment": log, "structures": stats}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
