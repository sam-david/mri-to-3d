# mri-to-3d

Turn the DICOM disc from an MRI or CT exam into 3D models of bones, organs, vessels and
muscles: one STL per structure, plus a single-file web viewer that works on desktop and phone.

```
mri3d build /Volumes/MRI_DISC -o models/lumbar --title "Lumbar MRI"
mri3d viewer models/lumbar -o viewer.html
```

Everything runs locally. Images never leave your machine; the only downloads are the
segmentation model weights (first run) and three.js, which the viewer loads from a CDN.

## Why another tool

Open-source pieces each do part of the job:

| Tool | Gap for clinical MRI |
|---|---|
| Horos / OsiriX volume rendering | On MRI, bone isn't bright as it is on CT, and thick slices render as stacked plates. No per-structure models. |
| InVesalius | Threshold-based surfaces, designed for CT. |
| 3D Slicer | Excellent but manual, one series at a time. |
| [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) | Accurate AI labels, but per series and stair-stepped on thick-slice MRI. |

Most clinical MRI isn't a 3D acquisition. It's several stacks of 3–7 mm slices in
different planes. `mri-to-3d` uses TotalSegmentator for the labels and adds what's missing:

1. **Series selection.** Picks the sharpest non-contrast anatomical series per plane and
   per study, and skips scouts, diffusion maps and screenshots (`mri3d inspect` shows the choice).
2. **Multi-plane fusion.** Each structure's mask from every series is softened, resampled
   into one isotropic world-space grid and averaged, with weights feathered toward each
   field-of-view edge. A sagittal stack contributes detail top-to-bottom and front-to-back;
   an axial stack contributes left-to-right.
3. **Cross-study alignment.** Exams can be re-zeroed between acquisitions (common for
   separate thoracic and lumbar studies, and always true across dates). Groups are aligned
   by translation onto structures they both contain completely, preferring rigid bone. The
   residual is reported.
4. **Spine niceties.** Per-level vertebrae, discs split and named by neighbours
   (`disc_L4-L5`), optional highlighting.
5. **Viewer.** Self-contained HTML: browse structures by category, see-through soft tissue,
   X-ray bones, click to identify, standard anatomical views, mobile layout.

## Install

Requires Python ≥ 3.10 and [dcm2niix](https://github.com/rordenlab/dcm2niix)
(`brew install dcm2niix` / `apt install dcm2niix`).

```
git clone https://github.com/sam-david/mri-to-3d && cd mri-to-3d
uv venv && uv pip install -e .        # or: python -m venv .venv && pip install -e .
```

The models run on NVIDIA GPUs, Apple Silicon (MPS) or CPU. On an M-series Mac a typical
series takes about a minute.

## Usage

```
mri3d inspect DICOM_DIR                       # list series, mark the ones build will use
mri3d build DICOM_DIR [DICOM_DIR ...] -o OUT  # convert -> segment -> fuse
          [--preset auto|spine|body|limbs|muscles] [--series REGEX]
          [--title T] [--date D] [--highlight vertebrae_L4 disc_L4-L5]
mri3d mark OUT --foramen L4-L5:left [--note "report finding"]   # mark a nerve-root exit
mri3d viewer OUT [OUT ...] -o viewer.html     # one file, every model inlined
mri3d viewer OUT [OUT ...] -o site --folder   # site/index.html + site/models/*.glb for hosting
```

Several DICOM folders passed to one `build` are aligned into a single model. For example,
separate cervical, thoracic and lumbar exams become a whole-spine model.

| Preset | MRI models | CT models | Covers |
|---|---|---|---|
| `auto` (default) | `total_mr`, `vertebrae_mr` | `total` | 50 MR / 117 CT structures + per-level vertebrae |
| `spine` | `vertebrae_mr`, `total_mr` | `total` | as above, spine labels take priority |
| `body` | `total_mr` | `total` | organs, vessels, major bones and muscles |
| `limbs` | + `appendicular_bones_mr`* | + `appendicular_bones`* | forearm, knee, lower-leg and foot bones |
| `muscles` | + `thigh_shoulder_muscles_mr`* | + `thigh_shoulder_muscles`* | rotator cuff, deltoid, thigh compartments |

\* needs TotalSegmentator's free non-commercial licence:
`totalseg_set_license -l <key>` ([request one](https://backend.totalsegmentator.com/license-academic/)).

### Markers

Nerve roots and foramina are a few millimetres across, usually below what clinical slices can
segment. When a report names a level and side ("left L4-5 foraminal stenosis"), `mri3d mark`
places a marker on the exit path of that nerve root. It uses the patient's own reconstructed
vertebrae, disc and spinal cord: it starts at the cord's edge at the disc level, angles
the way foramina run in that region, and centres the height in the bony gap between the two
vertebrae. The viewer shows a toggle per marker that flies the camera to it, along with any `--note`.
Markers show location, not severity.

### Outputs

```
OUT/stl/<structure>.stl    full-resolution meshes in scanner millimetres (3D printing, CAD, Blender)
OUT/manifest.json          volumes, categories, which series each structure came from, alignment log
```

## Limits

- **Slice thickness sets the ceiling.** Small features (nerve foramina, ligaments, cartilage)
  are below what 3–7 mm slices can show. Thin-slice 3D sequences and CT give much better models.
- Alignment is translation-only. That's right for re-zeroed exams in the same session, and an
  approximation across dates, since posture differs.
- Brain: only the whole brain is labelled. Detailed parcellation needs a brain-specific tool
  such as FastSurfer or SynthSeg.
- **Not a medical device.** The models are visualisations from automated segmentation. Use
  the original images and radiology reports for any clinical decision.

## Privacy

`.gitignore` excludes DICOM, NIfTI, meshes and generated viewers. Your data stays in
`work/` and your output folders. A generated viewer embeds models derived from your scans,
so host it somewhere private (see below) rather than on a public site.

### Viewing on a phone

- **Single file:** AirDrop or message `viewer.html` to yourself and open it in the browser.
- **Private hosting:** `mri3d viewer ... --folder` produces a static site you can serve behind
  authentication (e.g. Cloudflare Access, Tailscale Serve, a password-protected host).

## Development

```
uv pip install -e ".[dev,qa]" && pytest
```

Tests use synthetic phantoms (balls scanned as thick-slice series), not real imaging.

## License

MIT. TotalSegmentator and its model weights are under their own licences; see its repository.
