"""mri3d: DICOM -> fused 3D models -> web viewer.

  mri3d inspect  DICOM_DIR                 list series and which ones would be used
  mri3d build    DICOM_DIR [...] -o OUT    convert, segment, fuse  -> OUT/stl/*.stl + manifest.json
  mri3d mark     OUT --foramen L4-L5:left  add a location marker (e.g. a nerve-root exit) to a build
  mri3d xray     DICOM_DIR -o OUT          radiographs (CR/DX) -> image set for the viewer
  mri3d viewer   OUT [...] -o viewer.html  bundle one or more builds into the web viewer
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from . import dicom, markers, segment, web, xray
from .segment import PRESETS


def _work_dir(work: Path, dicom_dir: Path) -> Path:
    tag = hashlib.sha1(str(dicom_dir.resolve()).encode()).hexdigest()[:8]
    return work / f"{re.sub(r'[^A-Za-z0-9]+', '_', dicom_dir.name)[:40]}_{tag}"


def _fmt(s: dicom.Series) -> str:
    sp = s.spacing
    return (f"{s.plane[:3]:3} {s.weighting:5} {sp[0]:.2f}x{sp[1]:.2f}x{sp[2]:.1f} mm  "
            f"{s.shape[2]:3d} slices  {'+C ' if s.contrast else '   '}{s.description}")


def cmd_inspect(a):
    series = dicom.convert(Path(a.dicom), _work_dir(Path(a.work), Path(a.dicom)) / "nifti")
    chosen = {s.path for s in dicom.select(series)}
    for study in sorted({s.study for s in series}):
        print(f"\n{study}")
        for s in series:
            if s.study == study:
                print(("  * " if s.path in chosen else "    ") + _fmt(s))
    print("\n* = used by `build` (override with --series REGEX)")


def cmd_build(a):
    out = Path(a.out)
    groups, sources, names = [], [], []
    for d in map(Path, a.dicom):
        wd = _work_dir(Path(a.work), d)
        series = dicom.convert(d, wd / "nifti")
        picked = ([s for s in series if re.search(a.series, s.description, re.I)] if a.series
                  else dicom.select(series))
        if not picked:
            raise SystemExit(f"no usable series in {d} (try `mri3d inspect {d}`)")
        for study in sorted({s.study for s in picked}):  # one group per scanner frame
            g = {}
            for s in (s for s in picked if s.study == study):
                print(f"segmenting  {_fmt(s)}")
                g[s.path.name] = segment.run(s.path, wd / "seg" / s.path.name[:-7], s.modality, a.preset, a.device)
                sources.append(f"{s.description} ({s.plane}, {s.shape[2]} × {s.thickness:.1f} mm)")
            groups.append(g); names.append(f"\"{study}\"")
    from .fuse import build  # heavy imports after argument errors
    info = {"id": a.id or out.name, "title": a.title or out.name, "date": a.date,
            "sources": "; ".join(sources), "notes": a.notes, "highlight": a.highlight or []}
    m = build(groups, out, info, names)
    multi = sum(1 for v in m["structures"].values() if (v["multi_series_fraction"] or 0) >= 0.8)
    print(f"\n{len(m['structures'])} structures -> {out}/stl  ({multi} reconstructed from 2+ series)")
    for spec in a.foramen or []:
        print(f"marked {markers.add(out, spec)['label']}")


def cmd_mark(a):
    for spec in a.foramen:
        mk = markers.add(Path(a.out), spec, a.note)
        print(f"marked {mk['label']} in {a.out}")


def cmd_xray(a):
    m = xray.convert(Path(a.dicom), Path(a.out), {"id": a.id or Path(a.out).name,
                                                   "title": a.title or Path(a.out).name, "date": a.date})
    for im in m["images"]:
        print(f"  {im['date']}  {im['study']}  {im['view']}  {im['width']}x{im['height']}")
    print(f"{len(m['images'])} images -> {a.out}  (check for burned-in text before sharing)")


def cmd_viewer(a):
    dest = web.export([Path(p) for p in a.builds], Path(a.out), folder=a.folder, title=a.title)
    print(f"wrote {dest}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="mri3d", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default="work", help="cache for converted series and segmentations (default ./work)")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("inspect", help="list series in a DICOM folder")
    i.add_argument("dicom")
    i.set_defaults(fn=cmd_inspect)

    b = sub.add_parser("build", help="DICOM folder(s) -> 3D models")
    b.add_argument("dicom", nargs="+", help="one or more DICOM folders/discs; several are aligned into one model")
    b.add_argument("-o", "--out", required=True, help="output folder for this model")
    b.add_argument("--preset", default="auto", choices=sorted(PRESETS), help="which anatomy models to run (default auto)")
    b.add_argument("--series", help="regex on series description, instead of automatic selection")
    b.add_argument("--title"); b.add_argument("--date"); b.add_argument("--id"); b.add_argument("--notes")
    b.add_argument("--highlight", nargs="*", help="structure names to highlight, e.g. vertebrae_L4 disc_L4-L5")
    b.add_argument("--device", choices=["gpu", "mps", "cpu"], help="default: auto-detect")
    b.add_argument("--foramen", nargs="*", metavar="LEVEL:SIDE", help="mark neural foramina, e.g. L4-L5:left C6-C7:right")
    b.set_defaults(fn=cmd_build)

    k = sub.add_parser("mark", help="add location markers to an existing build")
    k.add_argument("out", help="a build folder (contains manifest.json and stl/)")
    k.add_argument("--foramen", nargs="+", required=True, metavar="LEVEL:SIDE", help="e.g. L4-L5:left")
    k.add_argument("--note", help="text shown with the marker in the viewer (e.g. a report finding)")
    k.set_defaults(fn=cmd_mark)

    x = sub.add_parser("xray", help="radiographs (CR/DX) -> image set for the viewer")
    x.add_argument("dicom"); x.add_argument("-o", "--out", required=True)
    x.add_argument("--title"); x.add_argument("--date"); x.add_argument("--id")
    x.set_defaults(fn=cmd_xray)

    v = sub.add_parser("viewer", help="bundle builds into the web viewer")
    v.add_argument("builds", nargs="+")
    v.add_argument("-o", "--out", default="viewer.html")
    v.add_argument("--folder", action="store_true", help="write OUT/index.html + OUT/models/*.glb for static hosting")
    v.add_argument("--title", default="3D Scan Models")
    v.set_defaults(fn=cmd_viewer)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
