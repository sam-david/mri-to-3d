"""DICOM -> NIfTI conversion and automatic series selection.

Clinical exams contain many series: scouts, several contrasts, diffusion maps, post-contrast
repeats. For 3D modelling we want the sharpest anatomical series in each slice direction,
per study, so they can be fused later.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nb
import numpy as np

SKIP = re.compile(
    r"local|scout|survey|scanogram|3-pl|tri-?pilot|cal\b|calibration|screen|save|"
    r"dwi|diffusion|adc|apparent|trace|exponential|perf|map\b|mip|subtract",
    re.I)
CONTRAST = re.compile(r"\+c\b|\+gad|post|gad|contrast|\bc\+", re.I)
PLANES = ("sagittal", "coronal", "axial")


@dataclass
class Series:
    path: Path
    description: str
    study: str
    modality: str
    plane: str
    spacing: tuple
    shape: tuple
    contrast: bool
    meta: dict = field(repr=False, default_factory=dict)

    @property
    def thickness(self) -> float:
        return max(self.spacing)

    @property
    def is_3d(self) -> bool:
        return self.thickness <= 1.6 and min(self.shape) >= 40

    @property
    def weighting(self) -> str:
        d = self.description.lower()
        for w in ("stir", "flair", "t2", "t1", "pd"):
            if w in d:
                return w
        return "?"

    def world_bounds(self):
        img = nb.load(self.path)
        s = np.array(img.shape[:3]) - 1
        corners = np.array([[a, b, c] for a in (0, s[0]) for b in (0, s[1]) for c in (0, s[2])])
        w = nb.affines.apply_affine(img.affine, corners)
        return w.min(0), w.max(0)


def convert(dicom_dir: Path, out_dir: Path) -> list[Series]:
    """Convert every series under dicom_dir (anonymised sidecars) and describe them."""
    if shutil.which("dcm2niix") is None:
        raise SystemExit("dcm2niix not found: install it (brew install dcm2niix / apt install dcm2niix)")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not any(out_dir.glob("*.nii.gz")):
        subprocess.run(["dcm2niix", "-z", "y", "-b", "y", "-ba", "y", "-f", "%s_%d",
                        "-o", str(out_dir), str(dicom_dir)], check=True, capture_output=True)
    return [s for f in sorted(out_dir.glob("*.nii.gz")) if (s := describe(f))]


def describe(nii: Path) -> Series | None:
    sidecar = nii.with_name(nii.name[:-7] + ".json")
    if not sidecar.exists():  # dcm2niix-derived volumes (e.g. *_Eq_1 resampled stacks)
        return None
    meta = json.loads(sidecar.read_text())
    img = nb.load(nii)
    if img.ndim < 3 or img.shape[2] < 2:
        return None
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    # Slice direction = the voxel axis with the largest spacing; its dominant world axis
    # (R, A, S) gives the plane.
    axis = int(np.argmax(zooms))
    normal = np.abs(img.affine[:3, axis])
    plane = PLANES[int(np.argmax(normal))]
    desc = meta.get("SeriesDescription") or meta.get("ProtocolName") or nii.name[:-7]
    return Series(path=nii, description=desc,
                  study=meta.get("StudyDescription") or meta.get("StudyInstanceUID") or "study",
                  modality=meta.get("Modality", "MR"), plane=plane, spacing=zooms,
                  shape=tuple(img.shape[:3]), contrast=bool(CONTRAST.search(desc)), meta=meta)


def score(s: Series, prefer: tuple[str, ...]) -> float:
    """Higher is better: sharp, thin, non-contrast, preferred weighting, 3D bonus."""
    voxel = s.spacing[0] * s.spacing[1] * s.thickness
    sc = -np.log(voxel) - 0.35 * s.thickness
    if s.is_3d:
        sc += 3
    if s.contrast:
        sc -= 1.5
    if s.weighting in prefer:
        sc += 3 - 1.5 * prefer.index(s.weighting)
    if re.search(r"t2\*|\bgre\b|fgre|flash|medic|merge", s.description, re.I) and not s.is_3d:
        sc -= 0.6  # gradient echo: more susceptibility blur than spin echo
    if s.weighting in ("stir", "flair"):
        sc -= 1
    return sc


def select(series: list[Series], prefer=("t2", "t1")) -> list[Series]:
    """Pick, per study and slice plane, the best series plus any same-plane series that
    covers a mostly different region (e.g. a separate thoracic and lumbar station)."""
    usable = [s for s in series if not SKIP.search(s.description) and min(s.shape) >= 5
              and s.modality in ("MR", "CT")]
    chosen = []
    for study in sorted({s.study for s in usable}):
        for plane in PLANES:
            cands = sorted((s for s in usable if s.study == study and s.plane == plane),
                           key=lambda s: -score(s, prefer))
            picked = []
            for c in cands:
                if c.contrast and any(not p.contrast for p in picked):
                    continue
                if all(_overlap(c, p) < 0.5 for p in picked) and (not picked or score(c, prefer) > score(picked[0], prefer) - 4):
                    picked.append(c)
            chosen += picked
    # Post-contrast repeats are usually filed as their own study; drop a study whose picks
    # are all post-contrast when a non-contrast study covers the same region.
    plain = [s for s in chosen if not s.contrast]
    chosen = [s for s in chosen if not s.contrast or not plain or
              any(not c.contrast for c in chosen if c.study == s.study) or
              all(_overlap(s, p) < 0.5 for p in plain)]
    # A true 3D series already covers every direction; drop weaker 2D series it contains.
    threes = [s for s in chosen if s.is_3d]
    if threes:
        chosen = threes + [s for s in chosen if not s.is_3d and all(_overlap(s, t) < 0.8 for t in threes)]
    return chosen


def _overlap(a: Series, b: Series) -> float:
    """Fraction of a's field of view (bounding box volume) inside b's."""
    alo, ahi = a.world_bounds()
    blo, bhi = b.world_bounds()
    inter = np.clip(np.minimum(ahi, bhi) - np.maximum(alo, blo), 0, None).prod()
    return float(inter / max(np.prod(ahi - alo), 1e-6))
