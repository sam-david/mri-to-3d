"""Synthetic checks for fusion and alignment (no real imaging needed).

A ball is "scanned" as two thick-slice series in different planes. Fusing them should recover
the true volume better than either series alone, and a second study shifted into another
scanner frame should be snapped back onto the first.
"""
import json

import nibabel as nb
import numpy as np
import pytest

from mri3d import fuse

R = 20.0  # mm
TRUE_CM3 = 4 / 3 * np.pi * R ** 3 / 1000


def _scan(tmp, name, center, thick_axis, label="vertebrae_L1", extra=None, shift=(0, 0, 0)):
    """Rasterise balls into a 0.6 mm x 0.6 mm x 5 mm series; slices along thick_axis."""
    zooms = [0.6, 0.6, 0.6]; zooms[thick_axis] = 5.0
    shape = [int(80 / z) for z in zooms]
    aff = np.diag([*zooms, 1.0]); aff[:3, 3] = np.array([-40, -40, -40]) + shift
    world = nb.affines.apply_affine(aff, np.indices(shape).reshape(3, -1).T)
    d = tmp / name / "seg"; d.mkdir(parents=True)
    for lab, c in [(label, center), *(extra or [])]:
        inside = np.linalg.norm(world - (np.array(c) + shift), axis=1) <= R
        nb.save(nb.Nifti1Image(inside.reshape(shape).astype(np.uint8), aff), d / f"{lab}.nii.gz")
    return d


def _volume(manifest, name="vertebrae_L1"):
    return manifest["structures"][name]["volume_cm3"]


def test_fusion_beats_single_series(tmp_path):
    sag = _scan(tmp_path, "sag", (0, 0, 0), thick_axis=0)
    ax = _scan(tmp_path, "ax", (0, 0, 0), thick_axis=2)
    single = fuse.build([{"sag": [sag]}], tmp_path / "single", {"id": "s"})
    fused = fuse.build([{"sag": [sag], "ax": [ax]}], tmp_path / "fused", {"id": "f"})
    err_single = abs(_volume(single) - TRUE_CM3) / TRUE_CM3
    err_fused = abs(_volume(fused) - TRUE_CM3) / TRUE_CM3
    assert err_fused < 0.08
    assert err_fused <= err_single
    assert fused["structures"]["vertebrae_L1"]["multi_series_fraction"] > 0.8
    assert (tmp_path / "fused" / "stl" / "vertebrae_L1.stl").exists()
    assert json.loads((tmp_path / "fused" / "manifest.json").read_text())["structures"]["vertebrae_L1"]["category"] == "bone"


def test_groups_in_different_frames_are_aligned(tmp_path):
    shift = (12.0, -7.0, 400.0)  # e.g. table re-landmarked between two exams
    extra = [("vertebrae_L2", (0, 0, -30))]
    a = _scan(tmp_path, "a", (0, 0, 15), thick_axis=0, extra=extra)
    b = _scan(tmp_path, "b", (0, 0, 15), thick_axis=0, extra=extra, shift=shift)
    m = fuse.build([{"a": [a]}, {"b": [b]}], tmp_path / "out", {"id": "x"})
    assert any("residual 0.0 mm" in line or "residual 0.1 mm" in line for line in m["alignment"])
    # Aligned copies overlap, so the fused ball keeps its true size instead of doubling.
    assert abs(_volume(m) - TRUE_CM3) / TRUE_CM3 < 0.1


def test_unrelated_group_is_skipped(tmp_path):
    a = _scan(tmp_path, "a", (0, 0, 0), thick_axis=0, label="vertebrae_L1")
    b = _scan(tmp_path, "b", (0, 0, 0), thick_axis=0, label="femur_left")
    m = fuse.build([{"a": [a]}, {"b": [b]}], tmp_path / "out", {"id": "x"})
    assert any("skipped" in line for line in m["alignment"])


@pytest.mark.parametrize("name,cat,label", [
    ("vertebrae_L4", "bone", "L4 vertebra"),
    ("disc_L4-L5", "disc", "L4–L5 disc"),
    ("kidney_left", "organ", "Left kidney"),
    ("iliac_artery_right", "vessel", "Right iliac artery"),
    ("gluteus_maximus_left", "muscle", "Left gluteus maximus"),
    ("scapula_left", "bone", "Left shoulder blade"),
])
def test_anatomy_labels(name, cat, label):
    from mri3d.anatomy import category, pretty
    assert category(name) == cat
    assert pretty(name) == label


@pytest.mark.parametrize("upper,lower,root", [
    ("C4", "C5", "C5"), ("C7", "T1", "C8"), ("T6", "T7", "T6"), ("L4", "L5", "L4"), ("L5", "S1", "L5"),
])
def test_exiting_root(upper, lower, root):
    from mri3d.markers import exiting_root
    assert exiting_root(upper, lower) == root


def test_xray_conversion(tmp_path):
    """A synthetic MONOCHROME1 radiograph is found, inverted (bone white) and exported."""
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    from mri3d import xray

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(tmp_path / "img"), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Modality, ds.StudyDescription, ds.ViewPosition, ds.StudyDate = "CR", "XR Phantom", "AP", "20250102"
    ds.Rows, ds.Columns, ds.SamplesPerPixel = 64, 48, 1
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    a = np.full((64, 48), 3000, np.uint16); a[20:40, 10:30] = 200  # "bone" is low in MONOCHROME1
    ds.PixelData = a.tobytes()
    (tmp_path / "dicom").mkdir(); ds.save_as(tmp_path / "dicom" / "IM0001", enforce_file_format=True)

    m = xray.convert(tmp_path / "dicom", tmp_path / "out", {"id": "x", "title": "t"})
    assert [(i["study"], i["view"], i["date"]) for i in m["images"]] == [("XR Phantom", "AP", "2025-01-02")]
    from PIL import Image
    px = np.asarray(Image.open(tmp_path / "out" / m["images"][0]["src"]))
    assert px[30, 20] > px[5, 5]  # inverted: the dense square renders brighter than background
