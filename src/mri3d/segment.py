"""Run TotalSegmentator models on selected series (locally; cached per series/task)."""
from __future__ import annotations

from pathlib import Path

# Which TotalSegmentator tasks to run for each region preset, per modality.
# Tasks marked * need a free non-commercial licence: `totalseg_set_license -l <key>`
# (https://backend.totalsegmentator.com/license-academic/).
PRESETS = {
    "auto":     {"MR": ["total_mr", "vertebrae_mr"], "CT": ["total"]},
    "spine":    {"MR": ["vertebrae_mr", "total_mr"], "CT": ["total"]},
    "body":     {"MR": ["total_mr"], "CT": ["total"]},
    "limbs":    {"MR": ["total_mr", "appendicular_bones_mr*"], "CT": ["total", "appendicular_bones*"]},
    "muscles":  {"MR": ["total_mr", "thigh_shoulder_muscles_mr*"], "CT": ["total", "thigh_shoulder_muscles*"]},
}


def device() -> str:
    import torch
    if torch.cuda.is_available():
        return "gpu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run(nii: Path, out_root: Path, modality: str, preset: str, dev: str | None = None) -> list[Path]:
    """Segment one series with every task of the preset. Returns the output folders."""
    from totalsegmentator.python_api import totalsegmentator

    tasks = PRESETS[preset]["CT" if modality == "CT" else "MR"]
    outs = []
    for t in tasks:
        task = t.rstrip("*")
        out = out_root / task
        if not (out.exists() and any(out.glob("*.nii.gz"))):
            try:
                totalsegmentator(str(nii), str(out), task=task, device=dev or device(), quiet=True)
            except Exception as e:  # e.g. licence missing for a * task: keep going with the rest
                print(f"  ! {task} skipped on {nii.name}: {str(e).splitlines()[0][:160]}")
                continue
        outs.append(out)
    return outs
