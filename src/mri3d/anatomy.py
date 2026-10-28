"""Structure categories, colours and display names for TotalSegmentator labels."""
import re

CATEGORY_COLORS = {  # sRGB
    "bone": "#e8dcc4",
    "disc": "#5aa0e6",
    "neural": "#f0c440",
    "organ": "#d98c8c",
    "lung": "#a8c8e0",
    "vessel": "#c0392b",
    "muscle": "#b5655a",
    "fat": "#f2d98d",
    "other": "#b0b0b8",
}

BONE = re.compile(r"^(vertebrae|sacrum|hip|femur|humerus|scapula|clavicula|rib|sternum|skull|"
                  r"patella|tibia|fibula|tarsal|metatarsal|phalanges|ulna|radius|carpal|"
                  r"metacarpal|costal|mandible)")
VESSEL = re.compile(r"(aorta|vena|artery|arteries|vein|iliac_|portal|carotid|subclavian|brachiocephalic)")
MUSCLE = re.compile(r"(gluteus|iliopsoas|autochthon|quadriceps|sartorius|deltoid|spinatus|subscapularis|"
                    r"coracobrachial|trapezius|pectoralis|serratus|teres|triceps|thigh_|muscle)")
ORGAN = re.compile(r"(liver|kidney|spleen|pancreas|gallbladder|stomach|bowel|duodenum|colon|"
                   r"esophagus|bladder|prostate|adrenal|heart|thyroid|trachea)")


def category(name: str) -> str:
    if name.startswith("disc_") or name == "intervertebral_discs":
        return "disc"
    if name in ("spinal_cord", "brain") or name.startswith("brain"):
        return "neural"
    if name.startswith("lung"):
        return "lung"
    if BONE.search(name):
        return "bone"
    if VESSEL.search(name):
        return "vessel"
    if MUSCLE.search(name):
        return "muscle"
    if ORGAN.search(name):
        return "organ"
    if "fat" in name:
        return "fat"
    return "other"


NICE = {"clavicula": "collarbone", "scapula": "shoulder blade", "humerus": "humerus (upper arm)",
        "femur": "femur (thigh bone)", "hip": "hip bone", "autochthon": "deep back muscles",
        "iliopsoas": "iliopsoas", "inferior_vena_cava": "inferior vena cava",
        "portal_vein_and_splenic_vein": "portal & splenic vein", "small_bowel": "small bowel"}


def pretty(name: str) -> str:
    if m := re.match(r"^vertebrae_(\w+)$", name):
        return f"{m[1]} vertebra"
    if m := re.match(r"^disc_(\w+)-(\w+)$", name):
        return f"{m[1]}–{m[2]} disc"
    side = ""
    for s in ("left", "right"):
        if name.endswith("_" + s):
            side, name = s.capitalize() + " ", name[: -len(s) - 1]
    base = NICE.get(name, name.replace("_", " "))
    return (side + base)[:1].upper() + (side + base)[1:]


SPINE_ORDER = {"C": 0, "T": 10, "L": 30, "S": 40}


def level(name: str) -> int:
    m = re.search(r"(?:vertebrae_|disc_)([CTLS])(\d+)", name)
    return SPINE_ORDER[m[1]] + int(m[2]) if m else 999
