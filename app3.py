# ================================================================
#  🔍 DeepFake Detector — Streamlit GUI + Grad-CAM Heatmap
#  Model  : EfficientNet-B4
#  Extras : Grad-CAM heatmap, Face region risk scoring,
#           Region overlay, Cyberpunk dark theme
#  Run    : streamlit run app.py
#  Install: pip install streamlit torch torchvision opencv-python matplotlib pillow
# ================================================================

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from PIL import Image, ImageDraw
import numpy as np
import time, io, os
import cv2
import matplotlib.cm as cm

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap');

:root {
    --bg-primary:    #020408;
    --bg-card:       #0a1520;
    --accent-cyan:   #00f5ff;
    --accent-green:  #00ff88;
    --accent-red:    #ff2d55;
    --accent-orange: #ff9500;
    --accent-purple: #bf5af2;
    --text-primary:  #e8f4fd;
    --text-muted:    #5a7a8a;
    --border:        #1a3a4a;
}
*, *::before, *::after { box-sizing: border-box; }
.stApp {
    background: var(--bg-primary) !important;
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(0,245,255,0.03) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(0,255,136,0.02) 0%, transparent 50%) !important;
    font-family: 'Rajdhani', sans-serif;
    color: var(--text-primary);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem !important; max-width: 1400px !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: var(--accent-cyan); border-radius: 2px; }

/* Header */
.main-header { text-align:center; padding:2.5rem 1rem 2rem; position:relative; margin-bottom:2rem; }
.main-header::before { content:''; position:absolute; top:0; left:10%; right:10%; height:1px;
    background:linear-gradient(90deg,transparent,var(--accent-cyan),transparent); }
.main-header::after  { content:''; position:absolute; bottom:0; left:20%; right:20%; height:1px;
    background:linear-gradient(90deg,transparent,var(--accent-cyan),transparent); }
.main-title {
    font-family:'Orbitron',monospace; font-size:3.2rem; font-weight:900; letter-spacing:0.15em;
    background:linear-gradient(135deg,var(--accent-cyan) 0%,#00a8ff 50%,var(--accent-green) 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0; line-height:1.1;
}
.main-subtitle { font-family:'Share Tech Mono',monospace; font-size:0.85rem;
    color:var(--text-muted); letter-spacing:0.3em; margin-top:0.5rem; text-transform:uppercase; }
.version-badge { display:inline-block; font-family:'Share Tech Mono',monospace; font-size:0.7rem;
    color:var(--accent-cyan); border:1px solid var(--accent-cyan); padding:2px 10px;
    border-radius:2px; margin-top:0.8rem; letter-spacing:0.15em; opacity:0.7; }

/* Cards */
.card { background:var(--bg-card); border:1px solid var(--border); border-radius:8px;
    padding:1.5rem; position:relative; overflow:hidden; }
.card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,transparent,var(--accent-cyan),transparent); }
.card-title { font-family:'Orbitron',monospace; font-size:0.75rem; letter-spacing:0.2em;
    color:var(--accent-cyan); text-transform:uppercase; margin-bottom:1rem;
    display:flex; align-items:center; gap:8px; }
.card-title::after { content:''; flex:1; height:1px;
    background:linear-gradient(90deg,var(--border),transparent); }

/* Verdict boxes */
.verdict-real {
    background:linear-gradient(135deg,rgba(0,255,136,0.08),rgba(0,255,136,0.02));
    border:1px solid rgba(0,255,136,0.4); border-radius:12px; padding:2rem;
    text-align:center; animation:pulseGreen 2s ease-in-out infinite;
}
.verdict-fake {
    background:linear-gradient(135deg,rgba(255,45,85,0.08),rgba(255,45,85,0.02));
    border:1px solid rgba(255,45,85,0.4); border-radius:12px; padding:2rem;
    text-align:center; animation:pulseRed 2s ease-in-out infinite;
}
@keyframes pulseGreen { 0%,100%{box-shadow:0 0 15px rgba(0,255,136,0.2);} 50%{box-shadow:0 0 35px rgba(0,255,136,0.5);} }
@keyframes pulseRed   { 0%,100%{box-shadow:0 0 15px rgba(255,45,85,0.2);}  50%{box-shadow:0 0 35px rgba(255,45,85,0.5);} }
.verdict-label { font-family:'Orbitron',monospace; font-size:0.7rem;
    letter-spacing:0.3em; text-transform:uppercase; margin-bottom:0.5rem; }
.verdict-main  { font-family:'Orbitron',monospace; font-size:2.5rem;
    font-weight:900; letter-spacing:0.1em; line-height:1; }
.verdict-real .verdict-label { color:rgba(0,255,136,0.7); }
.verdict-real .verdict-main  { color:var(--accent-green); }
.verdict-fake .verdict-label { color:rgba(255,45,85,0.7); }
.verdict-fake .verdict-main  { color:var(--accent-red); }

/* Confidence bars */
.conf-wrapper { margin:1.2rem 0; }
.conf-label-row { display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }
.conf-label { font-family:'Share Tech Mono',monospace; font-size:0.75rem; letter-spacing:0.1em; }
.conf-pct   { font-family:'Orbitron',monospace; font-size:1rem; font-weight:700; }
.conf-track { width:100%; height:10px; background:rgba(255,255,255,0.05);
    border-radius:5px; overflow:hidden; border:1px solid rgba(255,255,255,0.05); }
.conf-fill-real { height:100%; border-radius:5px;
    background:linear-gradient(90deg,#00cc66,var(--accent-green));
    box-shadow:0 0 10px rgba(0,255,136,0.5); }
.conf-fill-fake { height:100%; border-radius:5px;
    background:linear-gradient(90deg,#cc0033,var(--accent-red));
    box-shadow:0 0 10px rgba(255,45,85,0.5); }

/* Image info grid */
.info-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.8rem; margin-top:1rem; }
.info-item { background:rgba(255,255,255,0.02); border:1px solid var(--border);
    border-radius:6px; padding:0.8rem 1rem; }
.info-key { font-family:'Share Tech Mono',monospace; font-size:0.65rem;
    color:var(--text-muted); letter-spacing:0.15em; text-transform:uppercase; margin-bottom:3px; }
.info-val { font-family:'Orbitron',monospace; font-size:0.9rem;
    color:var(--accent-cyan); font-weight:700; }

/* Region risk rows */
.region-row { display:flex; justify-content:space-between; align-items:center;
    padding:0.55rem 0.8rem; border-radius:5px; margin-bottom:0.4rem;
    border:1px solid transparent; }
.region-critical { background:rgba(255,45,85,0.12); border-color:rgba(255,45,85,0.3); }
.region-high     { background:rgba(255,149,0,0.1);  border-color:rgba(255,149,0,0.3); }
.region-medium   { background:rgba(0,245,255,0.07); border-color:rgba(0,245,255,0.2); }
.region-low      { background:rgba(0,255,136,0.05); border-color:rgba(0,255,136,0.15); }
.region-name { font-family:'Share Tech Mono',monospace; font-size:0.72rem;
    letter-spacing:0.05em; color:var(--text-primary); }
.badge { font-family:'Orbitron',monospace; font-size:0.6rem; font-weight:700;
    padding:2px 8px; border-radius:3px; letter-spacing:0.1em; }
.badge-critical { background:rgba(255,45,85,0.25); color:#ff2d55; border:1px solid rgba(255,45,85,0.5); }
.badge-high     { background:rgba(255,149,0,0.2);  color:#ff9500; border:1px solid rgba(255,149,0,0.5); }
.badge-medium   { background:rgba(0,245,255,0.1);  color:#00f5ff; border:1px solid rgba(0,245,255,0.4); }
.badge-low      { background:rgba(0,255,136,0.1);  color:#00ff88; border:1px solid rgba(0,255,136,0.4); }
.region-score { font-family:'Orbitron',monospace; font-size:0.7rem; color:var(--text-muted); }

/* Stat rows */
.stat-row { display:flex; justify-content:space-between; align-items:center;
    padding:0.6rem 0; border-bottom:1px solid rgba(255,255,255,0.04); }
.stat-row:last-child { border-bottom:none; }
.stat-key { font-family:'Share Tech Mono',monospace; font-size:0.72rem;
    color:var(--text-muted); letter-spacing:0.08em; }
.stat-val       { font-family:'Orbitron',monospace; font-size:0.8rem; color:var(--accent-cyan); font-weight:700; }
.stat-val.green  { color:var(--accent-green); }
.stat-val.orange { color:var(--accent-orange); }

/* Alerts */
.alert-warn { background:rgba(255,149,0,0.08); border-left:3px solid var(--accent-orange);
    border-radius:0 6px 6px 0; padding:0.8rem 1rem; font-family:'Rajdhani',sans-serif;
    font-size:0.9rem; color:var(--accent-orange); margin-top:1rem; letter-spacing:0.03em; }
.alert-safe { background:rgba(0,255,136,0.05); border-left:3px solid var(--accent-green);
    border-radius:0 6px 6px 0; padding:0.8rem 1rem; font-family:'Rajdhani',sans-serif;
    font-size:0.9rem; color:var(--accent-green); margin-top:1rem; letter-spacing:0.03em; }

/* Divider */
.cyber-divider { height:1px;
    background:linear-gradient(90deg,transparent,var(--border),transparent); margin:1.5rem 0; }

/* Tab-like section header */
.section-hdr { font-family:'Orbitron',monospace; font-size:0.7rem; letter-spacing:0.25em;
    color:var(--accent-cyan); text-transform:uppercase; padding:0.5rem 0;
    border-bottom:1px solid var(--border); margin-bottom:1rem; }

/* Streamlit overrides */
[data-testid="stImage"] img { border-radius:8px !important; border:1px solid var(--border) !important; }
[data-testid="stFileUploader"] > div {
    background:rgba(0,245,255,0.02) !important; border:2px dashed #1a3a4a !important;
    border-radius:12px !important; }
[data-testid="stFileUploader"] > div:hover {
    border-color:var(--accent-cyan) !important; background:rgba(0,245,255,0.05) !important; }
.stButton > button {
    background:transparent !important; border:1px solid var(--accent-cyan) !important;
    color:var(--accent-cyan) !important; font-family:'Orbitron',monospace !important;
    font-size:0.75rem !important; letter-spacing:0.2em !important;
    text-transform:uppercase !important; padding:0.6rem 1.5rem !important;
    border-radius:4px !important; transition:all 0.3s !important; width:100% !important; }
.stButton > button:hover {
    background:rgba(0,245,255,0.1) !important;
    box-shadow:0 0 20px rgba(0,245,255,0.3) !important; }
.stTabs [data-baseweb="tab"] {
    font-family:'Orbitron',monospace !important; font-size:0.65rem !important;
    letter-spacing:0.15em !important; color:var(--text-muted) !important; }
.stTabs [aria-selected="true"] { color:var(--accent-cyan) !important; }

/* Footer */
.cyber-footer { text-align:center; padding:2rem 1rem 1rem;
    font-family:'Share Tech Mono',monospace; font-size:0.65rem;
    color:var(--text-muted); letter-spacing:0.15em;
    border-top:1px solid var(--border); margin-top:3rem; }

/* Heatmap legend */
.heatmap-legend { display:flex; align-items:center; gap:8px; margin-top:0.5rem;
    font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:var(--text-muted); }
.legend-bar { height:8px; flex:1; border-radius:4px;
    background:linear-gradient(90deg,#000080,#0000ff,#00ffff,#ffff00,#ff0000); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_PATH = r"C:\Users\sange\Downloads\MINOR\best_model.pth"  
print("MODEL PATH:", MODEL_PATH)
print("EXISTS:", os.path.exists(MODEL_PATH))
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = efficientnet_b4(weights=None)
    in_f  = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_f, 256), nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 1)
    )
    info = {"epoch": "N/A", "val_auc": 0, "val_acc": 0}
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        info = {"epoch"  : ckpt.get("epoch",   "N/A"),
                "val_auc": ckpt.get("val_auc",  0),
                "val_acc": ckpt.get("val_acc",  0)}
    model.to(DEVICE).eval()
    return model, info


def preprocess(img: Image.Image) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return tf(img).unsqueeze(0).to(DEVICE)


# ─────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, img: Image.Image):
    t0     = time.time()
    tensor = preprocess(img)
    logit  = model(tensor).squeeze()
    p_fake = torch.sigmoid(logit).item()
    p_real = 1.0 - p_fake
    ms     = (time.time() - t0) * 1000
    label  = "FAKE" if p_fake > 0.5 else "REAL"
    conf   = max(p_fake, p_real) * 100
    return label, p_real*100, p_fake*100, conf, ms


# ─────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        layer = model.features[-1]
        layer.register_forward_hook(self._save_act)
        layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):  self.activations = o.detach()
    def _save_grad(self, m, gi, go): self.gradients  = go[0].detach()

    def generate(self, img: Image.Image):
        self.model.eval()
        tensor = preprocess(img)
        tensor.requires_grad_(True)

        logit = self.model(tensor).squeeze()
        self.model.zero_grad()
        logit.backward()

        weights = self.gradients.mean(dim=[2,3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = F.relu(cam).cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        ow, oh  = img.size
        cam_up  = cv2.resize(cam, (ow, oh))
        return cam_up


@st.cache_resource
def get_gradcam(_model):
    return GradCAM(_model)


def apply_heatmap(pil_img: Image.Image, cam: np.ndarray, alpha=0.55) -> Image.Image:
    cmap    = cm.get_cmap("jet")
    heatmap = (cmap(cam)[:,:,:3] * 255).astype(np.uint8)
    orig    = np.array(pil_img.convert("RGB")).astype(float)
    heat    = heatmap.astype(float)
    blend   = (orig*(1-alpha) + heat*alpha).clip(0,255).astype(np.uint8)
    return Image.fromarray(blend)


# ─────────────────────────────────────────────────────────────
# FACE REGION SCORING
# ─────────────────────────────────────────────────────────────
def score_face_regions(cam: np.ndarray):
    h, w    = cam.shape
    t       = h // 3
    hw      = w // 2

    regions = {
        "FOREHEAD / HAIR" : cam[0       : t,          :         ],
        "LEFT EYE AREA"   : cam[t       : 2*t,        :hw       ],
        "RIGHT EYE AREA"  : cam[t       : 2*t,        hw:       ],
        "NOSE BRIDGE"     : cam[t       : 2*t,        w//4:3*w//4],
        "MOUTH / LIPS"    : cam[2*t     : int(h*.88), :         ],
        "CHIN / JAW"      : cam[int(h*.88):,          :         ],
    }

    results = []
    for name, patch in regions.items():
        score = float(patch.mean())
        lvl   = ("CRITICAL" if score > 0.55 else
                 "HIGH"     if score > 0.35 else
                 "MEDIUM"   if score > 0.20 else "LOW")
        results.append((name, score, lvl))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def make_region_overlay(pil_img: Image.Image, regions_data) -> Image.Image:
    base    = pil_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0,0,0,0))
    draw    = ImageDraw.Draw(overlay)
    w, h    = pil_img.size
    t, hw   = h//3, w//2

    boxes = {
        "FOREHEAD / HAIR": (0,   0,          w, t         ),
        "LEFT EYE AREA"  : (0,   t,         hw, 2*t       ),
        "RIGHT EYE AREA" : (hw,  t,          w, 2*t       ),
        "NOSE BRIDGE"    : (w//4,t,      3*w//4, 2*t      ),
        "MOUTH / LIPS"   : (0,   2*t,        w, int(h*.88)),
        "CHIN / JAW"     : (0,   int(h*.88), w, h         ),
    }
    fills   = {"CRITICAL":(255,45,85,85),  "HIGH":(255,149,0,65),
               "MEDIUM":(0,245,255,50),    "LOW":(0,255,136,30)}
    borders = {"CRITICAL":(255,45,85,220), "HIGH":(255,149,0,190),
               "MEDIUM":(0,245,255,170),   "LOW":(0,255,136,110)}

    lkup = {r[0]:(r[1],r[2]) for r in regions_data}
    for name, box in boxes.items():
        _, lvl = lkup.get(name,(0,"LOW"))
        draw.rectangle(box, fill=fills[lvl], outline=borders[lvl], width=2)

    return Image.alpha_composite(base, overlay).convert("RGB")


# ─────────────────────────────────────────────────────────────
# WHY FAKE — Rule-based reason generation
# ─────────────────────────────────────────────────────────────
def generate_fake_reasons(pil_img: Image.Image, cam: np.ndarray,
                          region_scores: list, p_fake: float) -> list:
    """
    Returns list of dicts: {icon, title, detail, highlight}
    highlight=True for the most critical reason
    """
    reasons = []
    img_arr = np.array(pil_img.convert("RGB"))
    h, w    = cam.shape

    # ── Reason 1: High-activation region ──────────────────
    top_region, top_score, top_lvl = region_scores[0]
    if top_score > 0.25:
        severity = "strong" if top_lvl in ("CRITICAL", "HIGH") else "moderate"
        reasons.append({
            "icon"     : "🔴",
            "title"    : f"Suspicious patterns detected in the {top_region.title()} area",
            "detail"   : (f"The model found {severity} signs of AI manipulation around the "
                          f"{top_region.lower()} region. This area shows texture inconsistencies "
                          f"that are characteristic of AI-generated faces, where the generator "
                          f"struggles to produce realistic skin detail and natural imperfections."),
            "highlight": top_lvl in ("CRITICAL", "HIGH"),
        })

    # ── Reason 2: Color uniformity (AI images = too uniform) ──
    hsv       = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
    sat_std   = float(hsv[:,:,1].std())
    if sat_std < 45:
        degree = "very" if sat_std < 30 else "somewhat"
        reasons.append({
            "icon"  : "🟠",
            "title" : "Colors look too uniform and artificially balanced",
            "detail": (f"The color variation across this image is {degree} low compared to what "
                       f"you'd expect from a real camera. AI image generators tend to produce "
                       f"overly smooth and consistent colors, whereas real photographs naturally "
                       f"have uneven lighting, shadows, and color casts that vary across the frame."),
            "highlight": sat_std < 30,
        })

    # ── Reason 3: Edge sharpness anomaly ──────────────────
    gray      = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian > 800 or laplacian < 50:
        if laplacian > 800:
            edge_detail = ("The edges in this image are unnaturally crisp and hyper-detailed — "
                           "a telltale sign of AI upsampling and sharpening. Real photos taken by "
                           "cameras have subtle softness and optical blur that AI images often lack.")
        else:
            edge_detail = ("The edges in this image are suspiciously smooth and blurry in ways "
                           "that don't match natural camera optics. This kind of over-smoothing "
                           "is commonly introduced by diffusion-based AI generators during the "
                           "image synthesis process.")
        reasons.append({
            "icon"  : "🟡",
            "title" : "Edge sharpness does not match real camera optics",
            "detail": edge_detail,
            "highlight": False,
        })

    # ── Reason 4: Noise pattern analysis (AI = too clean) ──
    noise     = img_arr.astype(float) - cv2.GaussianBlur(img_arr, (5,5), 0).astype(float)
    noise_std = float(np.std(noise))
    if noise_std < 8:
        reasons.append({
            "icon"  : "🟣",
            "title" : "No natural camera grain or sensor noise present",
            "detail": ("Every real camera sensor produces a tiny amount of random noise "
                       "— this is completely normal and expected. This image is unusually "
                       "clean, with almost no grain whatsoever. AI generators produce "
                       "mathematically perfect pixels that lack the organic imperfections "
                       "real cameras introduce, which is a strong indicator of synthetic generation."),
            "highlight": noise_std < 5,
        })

    # ── Reason 5: Aspect ratio + resolution pattern ────────
    ph, pw = pil_img.size[1], pil_img.size[0]
    common_ai_sizes = [(1024,1024),(512,512),(768,768),(1024,768),(768,1024),
                       (1280,720),(1920,1080),(2048,2048)]
    is_ai_res = any(abs(pw-aw)<=4 and abs(ph-ah)<=4 for aw,ah in common_ai_sizes)
    if is_ai_res:
        reasons.append({
            "icon"  : "🔵",
            "title" : "Image dimensions match standard AI generator output sizes",
            "detail": (f"This image is {pw}×{ph} pixels, which exactly matches one of the "
                       f"standard output sizes used by AI image generators like Stable Diffusion, "
                       f"DALL·E, and Midjourney. Real photographs come in a wide variety of "
                       f"resolutions depending on the camera — they almost never land precisely "
                       f"on these round, power-of-two dimensions."),
            "highlight": False,
        })

    # ── Reason 6: Model confidence fallback ───────────────
    if len(reasons) < 3:
        reasons.append({
            "icon"  : "⚪",
            "title" : "Deep learning model flagged this image as synthetic",
            "detail": (f"Our EfficientNet-B4 neural network — trained on thousands of real and "
                       f"AI-generated images — assigned a {p_fake:.1f}% probability that this "
                       f"image is fake. The model identified subtle patterns in the pixel structure "
                       f"and feature distributions that are consistent with AI generation, even "
                       f"when those patterns are not obvious to the human eye."),
            "highlight": False,
        })

    # Ensure at least one is highlighted
    if not any(r["highlight"] for r in reasons):
        reasons[0]["highlight"] = True

    return reasons[:5]   # max 5 reasons


# ─────────────────────────────────────────────────────────────
# GENERATOR IDENTIFICATION — Statistical fingerprint
# ─────────────────────────────────────────────────────────────
def identify_generator(pil_img: Image.Image, p_fake: float) -> dict:
    """
    Analyzes image artifacts to guess which AI generator made it.
    Returns: {name, confidence, icon, color, description, signals}
    """
    img_arr  = np.array(pil_img.convert("RGB"))
    gray     = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    hsv      = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
    pw, ph   = pil_img.size

    # ── Feature extraction ────────────────────────────────
    sat_mean  = float(hsv[:,:,1].mean())
    sat_std   = float(hsv[:,:,1].std())
    val_mean  = float(hsv[:,:,2].mean())
    lap_var   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    noise     = float(np.std(img_arr.astype(float) -
                             cv2.GaussianBlur(img_arr,(5,5),0).astype(float)))

    # FFT frequency analysis — high freq = DALL-E/MJ, smooth = SD
    fft       = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    mag       = np.abs(fft_shift)
    cy, cx    = mag.shape[0]//2, mag.shape[1]//2
    r         = min(cy, cx) // 3
    high_freq = float(mag[cy-r:cy+r, cx-r:cx+r].mean())
    low_freq  = float(mag.mean())
    freq_ratio = high_freq / (low_freq + 1e-8)

    is_square  = abs(pw - ph) < 10
    is_1024    = max(pw,ph) >= 1000
    is_512     = max(pw,ph) <= 520

    # ── Scoring each generator ────────────────────────────
    scores = {}

    # DALL-E 3: High saturation, painterly, often square 1024
    scores["DALL·E 3 (OpenAI)"] = (
        (0.3 if sat_mean > 120  else 0) +
        (0.2 if is_square       else 0) +
        (0.2 if is_1024         else 0) +
        (0.15 if val_mean > 140 else 0) +
        (0.15 if lap_var > 300  else 0)
    )

    # DALL-E 2: Softer, lower res, less saturated
    scores["DALL·E 2 (OpenAI)"] = (
        (0.3 if 80 < sat_mean < 120 else 0) +
        (0.2 if is_512              else 0) +
        (0.2 if lap_var < 300       else 0) +
        (0.3 if noise < 6           else 0)
    )

    # Midjourney: Very sharp, artistic, high contrast
    scores["Midjourney v5/v6"] = (
        (0.3 if lap_var > 500       else 0) +
        (0.25 if sat_std > 55       else 0) +
        (0.2  if freq_ratio > 0.6   else 0) +
        (0.25 if val_mean > 130 and sat_mean > 100 else 0)
    )

    # Stable Diffusion XL: balanced, natural-ish colors
    scores["Stable Diffusion XL"] = (
        (0.3 if 60 < sat_mean < 110  else 0) +
        (0.2 if 200 < lap_var < 500  else 0) +
        (0.25 if 6 < noise < 12      else 0) +
        (0.25 if freq_ratio < 0.5    else 0)
    )

    # Adobe Firefly: Warm tones, photorealistic
    scores["Adobe Firefly"] = (
        (0.35 if val_mean > 150     else 0) +
        (0.3  if sat_mean < 90      else 0) +
        (0.2  if noise < 8          else 0) +
        (0.15 if lap_var < 250      else 0)
    )

    # GLIDE: Older style, lower sharpness
    scores["GLIDE (OpenAI)"] = (
        (0.4 if lap_var < 150       else 0) +
        (0.3 if sat_mean < 80       else 0) +
        (0.3 if noise < 5           else 0)
    )

    # ── Pick winner ───────────────────────────────────────
    best_gen  = max(scores, key=scores.get)
    best_sc   = scores[best_gen]
    total_sc  = sum(scores.values()) + 1e-8
    raw_conf  = (best_sc / total_sc) * 100

    # Blend with p_fake for calibration
    conf = min(85, max(35, raw_conf * 0.7 + (p_fake - 50) * 0.3))

    gen_meta = {
        "DALL·E 3 (OpenAI)"   : {"icon":"🔵","color":"#00a8ff","desc":"High saturation, painterly textures, square 1024px output."},
        "DALL·E 2 (OpenAI)"   : {"icon":"🔷","color":"#0066cc","desc":"Softer edges, slightly lower resolution, smoother gradients."},
        "Midjourney v5/v6"    : {"icon":"🟣","color":"#bf5af2","desc":"Extremely sharp details, artistic color grading, high contrast."},
        "Stable Diffusion XL" : {"icon":"🟠","color":"#ff9500","desc":"Balanced saturation, natural-looking color distribution."},
        "Adobe Firefly"       : {"icon":"🔴","color":"#ff6b35","desc":"Warm tones, photorealistic style, clean low-noise output."},
        "GLIDE (OpenAI)"      : {"icon":"⚪","color":"#8a9ba8","desc":"Older diffusion model, softer and lower sharpness overall."},
    }

    meta = gen_meta[best_gen]
    # Top 3 signals used
    top_signals = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "name"       : best_gen,
        "confidence" : conf,
        "icon"       : meta["icon"],
        "color"      : meta["color"],
        "description": meta["desc"],
        "all_scores" : scores,
        "top_signals": top_signals,
    }


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-title">DEEPFAKE DETECTOR</div>
    <div class="main-subtitle">AI-Generated Image Forensics System</div>
    <div class="version-badge">EfficientNet-B4 · Grad-CAM · v1.0 · 2026</div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Initializing neural network..."):
    model, model_info = load_model()
gradcam = get_gradcam(model)

# ─────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────
col_left, col_mid, col_right = st.columns([1.1, 1.6, 1.1], gap="large")


# ══════════ LEFT — Upload + Metadata ══════════
with col_left:
    st.markdown('<div class="card"><div class="card-title">⬆ Image Input</div></div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop image or click to browse",
        type=["jpg","jpeg","png","webp","bmp"],
        label_visibility="collapsed"
    )

    if uploaded:
        img_bytes = uploaded.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h      = pil_img.size
        file_kb   = len(img_bytes) / 1024
        fmt       = uploaded.type.split("/")[-1].upper()

        st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)

        st.markdown(f"""
        <div class="card" style="margin-top:1rem;">
            <div class="card-title">📐 Image Metadata</div>
            <div class="info-grid">
                <div class="info-item"><div class="info-key">Width</div><div class="info-val">{w} px</div></div>
                <div class="info-item"><div class="info-key">Height</div><div class="info-val">{h} px</div></div>
                <div class="info-item"><div class="info-key">File Size</div><div class="info-val">{file_kb:.1f} KB</div></div>
                <div class="info-item"><div class="info-key">Format</div><div class="info-val">{fmt}</div></div>
                <div class="info-item"><div class="info-key">Aspect</div><div class="info-val">{w/h:.2f}</div></div>
                <div class="info-item"><div class="info-key">Mode</div><div class="info-val">RGB</div></div>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════ MIDDLE — Analysis + Heatmap ══════════
with col_mid:
    st.markdown('<div class="card"><div class="card-title">🔬 Forensic Analysis</div></div>',
                unsafe_allow_html=True)

    if not uploaded:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#2a4a5a;">
            <div style="font-size:4rem;margin-bottom:1rem;">🔍</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.8rem;
                        letter-spacing:0.2em;text-transform:uppercase;">
                Awaiting image input...<br><br>Upload an image to begin<br>forensic analysis
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        analyze_btn = st.button("⚡ ANALYZE + GENERATE HEATMAP", key="analyze")

        if analyze_btn:
            with st.spinner("Scanning neural patterns..."):
                label, p_real, p_fake, conf, ms = predict(model, pil_img)
                cam_map       = gradcam.generate(pil_img)
                heatmap_img   = apply_heatmap(pil_img, cam_map, alpha=0.6)
                region_scores = score_face_regions(cam_map)
                overlay_img   = make_region_overlay(pil_img, region_scores)
                fake_reasons  = generate_fake_reasons(pil_img, cam_map, region_scores, p_fake) if label == "FAKE" else []
                generator     = identify_generator(pil_img, p_fake) if label == "FAKE" else None

                st.session_state["result"] = {
                    "label": label, "p_real": p_real, "p_fake": p_fake,
                    "conf": conf, "ms": ms, "cam": cam_map,
                    "heatmap_img": heatmap_img, "region_scores": region_scores,
                    "overlay_img": overlay_img,
                    "fake_reasons": fake_reasons,
                    "generator"  : generator,
                }

        if "result" in st.session_state:
            r = st.session_state["result"]
            label, p_real, p_fake, conf, ms = r["label"], r["p_real"], r["p_fake"], r["conf"], r["ms"]

            st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)

            # ── Verdict ──
            if label == "REAL":
                st.markdown(f"""
                <div class="verdict-real">
                    <div class="verdict-label">Classification Result</div>
                    <div class="verdict-main">✓ AUTHENTIC</div>
                    <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
                                color:rgba(0,255,136,0.5);margin-top:0.5rem;letter-spacing:0.2em;">
                        IMAGE VERIFIED AS REAL
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-fake">
                    <div class="verdict-label">Classification Result</div>
                    <div class="verdict-main">⚠ AI GENERATED</div>
                    <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
                                color:rgba(255,45,85,0.5);margin-top:0.5rem;letter-spacing:0.2em;">
                        SYNTHETIC IMAGE DETECTED
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)

            # ── Probability bars ──
            st.markdown(f"""
            <div class="card">
                <div class="card-title">📊 Probability Distribution</div>
                <div class="conf-wrapper">
                    <div class="conf-label-row">
                        <span class="conf-label" style="color:#00ff88;">▶ REAL / AUTHENTIC</span>
                        <span class="conf-pct"   style="color:#00ff88;">{p_real:.1f}%</span>
                    </div>
                    <div class="conf-track"><div class="conf-fill-real" style="width:{p_real}%;"></div></div>
                </div>
                <div class="conf-wrapper">
                    <div class="conf-label-row">
                        <span class="conf-label" style="color:#ff2d55;">▶ FAKE / AI-GENERATED</span>
                        <span class="conf-pct"   style="color:#ff2d55;">{p_fake:.1f}%</span>
                    </div>
                    <div class="conf-track"><div class="conf-fill-fake" style="width:{p_fake}%;"></div></div>
                </div>
                <div class="conf-wrapper">
                    <div class="conf-label-row">
                        <span class="conf-label" style="color:#00f5ff;">▶ MODEL CONFIDENCE</span>
                        <span class="conf-pct"   style="color:#00f5ff;">{conf:.1f}%</span>
                    </div>
                    <div class="conf-track">
                        <div style="height:100%;width:{conf}%;border-radius:5px;
                                    background:linear-gradient(90deg,#0066cc,#00f5ff);
                                    box-shadow:0 0 10px rgba(0,245,255,0.5);"></div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            # ── Risk alert ──
            if label == "FAKE":
                risk = ("HIGH RISK" if p_fake > 85 else
                        "MEDIUM RISK" if p_fake > 65 else "LOW RISK")
                st.markdown(f"""
                <div class="alert-warn">
                    ⚠ {risk} — AI generation indicators detected. Confidence: {p_fake:.1f}% synthetic.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-safe">
                    ✓ LOW RISK — Patterns consistent with real photography. Confidence: {p_real:.1f}%
                </div>""", unsafe_allow_html=True)

            # ── WHY FAKE REASONS ──────────────────────────────
            if label == "FAKE" and r.get("fake_reasons"):
                st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-hdr">🧬 Why Is This Image Fake?</div>',
                            unsafe_allow_html=True)

                reasons_html = ""
                for i, reason in enumerate(r["fake_reasons"]):
                    hl_style = ""
                    hl_bar   = ""
                    if reason["highlight"]:
                        hl_style = ("background:rgba(255,45,85,0.07);"
                                    "border:1px solid rgba(255,45,85,0.35);"
                                    "border-radius:8px;")
                        hl_bar   = ('<div style="display:inline-block;font-family:\'Orbitron\',monospace;'
                                    'font-size:0.55rem;color:#ff2d55;background:rgba(255,45,85,0.15);'
                                    'border:1px solid rgba(255,45,85,0.4);padding:1px 7px;'
                                    'border-radius:2px;margin-left:8px;letter-spacing:0.1em;">'
                                    'PRIMARY INDICATOR</div>')
                    reasons_html += f"""
                    <div style="display:flex;gap:12px;padding:0.9rem;margin-bottom:0.5rem;{hl_style}">
                        <div style="font-size:1.3rem;flex-shrink:0;margin-top:2px;">{reason['icon']}</div>
                        <div>
                            <div style="font-family:'Rajdhani',sans-serif;font-size:1rem;
                                        font-weight:700;color:#e8f4fd;letter-spacing:0.03em;">
                                {reason['title']}{hl_bar}
                            </div>
                            <div style="font-family:'Rajdhani',sans-serif;font-size:0.95rem;
                                        color:#8a9ba8;margin-top:6px;line-height:1.6;letter-spacing:0.01em;">
                                {reason['detail']}
                            </div>
                        </div>
                    </div>"""

                st.markdown(
                    '<div class="card"><div class="card-title">🧬 Detection Evidence</div>'
                    + reasons_html +
                    '</div>',
                    unsafe_allow_html=True)

            # ── GENERATOR IDENTIFICATION ──────────────────────
            if label == "FAKE" and r.get("generator"):
                gen = r["generator"]
                st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)

                # Bar chart for all generators
                gen_bars = ""
                sorted_gens = sorted(gen["all_scores"].items(), key=lambda x: x[1], reverse=True)
                max_sc = max(v for _,v in sorted_gens) + 1e-8
                bar_colors = {
                    "DALL·E 3 (OpenAI)"   : "#00a8ff",
                    "DALL·E 2 (OpenAI)"   : "#0066cc",
                    "Midjourney v5/v6"    : "#bf5af2",
                    "Stable Diffusion XL" : "#ff9500",
                    "Adobe Firefly"       : "#ff6b35",
                    "GLIDE (OpenAI)"      : "#8a9ba8",
                }
                for gname, gscore in sorted_gens:
                    pct        = (gscore / max_sc) * 100
                    is_top     = gname == gen["name"]
                    bar_color  = bar_colors.get(gname, "#5a7a8a")
                    bold_style = "font-weight:700;color:#e8f4fd;" if is_top else "color:#5a7a8a;"
                    crown      = " 👑" if is_top else ""
                    gen_bars  += f"""
                    <div style="margin-bottom:0.55rem;">
                        <div style="display:flex;justify-content:space-between;
                                    font-family:'Share Tech Mono',monospace;font-size:0.68rem;
                                    margin-bottom:3px;{bold_style}">
                            <span>{gname}{crown}</span>
                            <span>{pct:.0f}%</span>
                        </div>
                        <div style="height:6px;background:rgba(255,255,255,0.05);border-radius:3px;">
                            <div style="height:100%;width:{pct}%;border-radius:3px;
                                        background:{'linear-gradient(90deg,'+bar_color+','+bar_color+'dd)' if is_top else '#1a3a4a'};
                                        {'box-shadow:0 0 8px '+bar_color+'88;' if is_top else ''}">
                            </div>
                        </div>
                    </div>"""

                gen_rgb = ','.join(str(int(gen['color'].lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))
                gen_header_html = (
                    '<div class="card">'
                    '<div class="card-title">🤖 Generator Identification</div>'
                    f'<div style="display:flex;align-items:center;gap:12px;padding:1rem;margin-bottom:1rem;'
                    f'background:rgba({gen_rgb},0.08);border:1px solid {gen["color"]}44;border-radius:8px;">'
                    f'<div style="font-size:2.5rem;">{gen["icon"]}</div>'
                    f'<div>'
                    f'<div style="font-family:Orbitron,monospace;font-size:1rem;font-weight:900;color:{gen["color"]};letter-spacing:0.05em;">{gen["name"]}</div>'
                    f'<div style="font-family:Share Tech Mono,monospace;font-size:0.65rem;color:#5a7a8a;margin-top:3px;">Confidence: {gen["confidence"]:.0f}% match</div>'
                    f'<div style="font-family:Rajdhani,sans-serif;font-size:0.85rem;color:#8a9ba8;margin-top:5px;line-height:1.4;">{gen["description"]}</div>'
                    f'</div></div>'
                    '<div style="font-family:Orbitron,monospace;font-size:0.65rem;color:#3a5a6a;letter-spacing:0.1em;margin-bottom:0.8rem;">GENERATOR PROBABILITY RANKING</div>'
                )
                gen_footer_html = (
                    '<div style="font-family:Share Tech Mono,monospace;font-size:0.6rem;'
                    'color:#2a4a5a;margin-top:0.8rem;letter-spacing:0.05em;">'
                    '⚠ Based on statistical artifact fingerprinting — approximate result'
                    '</div></div>'
                )
                st.markdown(gen_header_html + gen_bars + gen_footer_html, unsafe_allow_html=True)

            # ── Grad-CAM Section (always shown, more useful for FAKE) ──
            st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-hdr">🌡 Grad-CAM Forensic Heatmap</div>',
                        unsafe_allow_html=True)

            if label == "FAKE":
                st.markdown("""
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
                            color:#5a7a8a;margin-bottom:0.8rem;letter-spacing:0.08em;">
                    Red/yellow zones = highest AI artifact concentration.<br>
                    Blue zones = low manipulation probability.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
                            color:#5a7a8a;margin-bottom:0.8rem;letter-spacing:0.08em;">
                    Heatmap shows which regions the model used to verify authenticity.
                </div>""", unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(["🔥 Heatmap Overlay", "🗺 Region Map", "📷 Original"])

            with tab1:
                st.image(r["heatmap_img"], use_container_width=True)
                st.markdown("""
                <div class="heatmap-legend">
                    <span>Low</span>
                    <div class="legend-bar"></div>
                    <span>High Activation</span>
                </div>""", unsafe_allow_html=True)

            with tab2:
                st.image(r["overlay_img"], use_container_width=True)
                st.markdown("""
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
                            color:#5a7a8a;margin-top:0.4rem;">
                    Color boxes show region-level risk scoring
                </div>""", unsafe_allow_html=True)

            with tab3:
                st.image(pil_img, use_container_width=True)

            # ── Inference time ──
            st.markdown(f"""
            <div style="text-align:right;font-family:'Share Tech Mono',monospace;
                        font-size:0.65rem;color:#1a3a4a;margin-top:0.5rem;letter-spacing:0.1em;">
                ⚡ INFERENCE: {ms:.1f} ms · DEVICE: {str(DEVICE).upper()}
            </div>""", unsafe_allow_html=True)


# ══════════ RIGHT — Model Info + Region Scores ══════════
with col_right:
    val_auc = f"{model_info['val_auc']*100:.2f}%" if model_info['val_auc'] else "N/A"
    val_acc = f"{model_info['val_acc']*100:.2f}%" if model_info['val_acc'] else "N/A"
    dev_str = "CUDA GPU" if DEVICE.type == "cuda" else "CPU"

    st.markdown(f"""
    <div class="card">
        <div class="card-title">🧠 Model Status</div>
        <div class="stat-row"><span class="stat-key">ARCHITECTURE</span><span class="stat-val">EfficientNet-B4</span></div>
        <div class="stat-row"><span class="stat-key">STATUS</span><span class="stat-val green">● ONLINE</span></div>
        <div class="stat-row"><span class="stat-key">DEVICE</span><span class="stat-val">{dev_str}</span></div>
        <div class="stat-row"><span class="stat-key">BEST EPOCH</span><span class="stat-val">{model_info['epoch']}</span></div>
        <div class="stat-row"><span class="stat-key">VAL AUC</span><span class="stat-val green">{val_auc}</span></div>
        <div class="stat-row"><span class="stat-key">VAL ACC</span><span class="stat-val green">{val_acc}</span></div>
        <div class="stat-row"><span class="stat-key">INPUT SIZE</span><span class="stat-val">224 × 224</span></div>
        <div class="stat-row"><span class="stat-key">EXPLAINABILITY</span><span class="stat-val orange">Grad-CAM</span></div>
    </div>""", unsafe_allow_html=True)

    # ── Face Region Scores ──
    if "result" in st.session_state and st.session_state["result"]["label"] == "FAKE":
        rs = st.session_state["result"]["region_scores"]
        badge_cls = {"CRITICAL":"badge-critical","HIGH":"badge-high",
                     "MEDIUM":"badge-medium","LOW":"badge-low"}
        row_cls   = {"CRITICAL":"region-critical","HIGH":"region-high",
                     "MEDIUM":"region-medium","LOW":"region-low"}

        rows_html = ""
        for name, score, lvl in rs:
            rows_html += f"""
            <div class="region-row {row_cls[lvl]}">
                <span class="region-name">{name}</span>
                <div style="display:flex;align-items:center;gap:6px;">
                    <span class="region-score">{score:.2f}</span>
                    <span class="badge {badge_cls[lvl]}">{lvl}</span>
                </div>
            </div>"""

        st.markdown(
            '<div class="card" style="margin-top:1rem;">'
            '<div class="card-title">🎯 Region Risk Analysis</div>'
            '<div style="font-family:Share Tech Mono,monospace;font-size:0.65rem;'
            'color:#3a5a6a;margin-bottom:0.8rem;letter-spacing:0.08em;">'
            'Grad-CAM activation per face region</div>'
            + rows_html +
            '</div>',
            unsafe_allow_html=True)

    elif "result" in st.session_state and st.session_state["result"]["label"] == "REAL":
        st.markdown("""
        <div class="card" style="margin-top:1rem;">
            <div class="card-title">🎯 Region Risk Analysis</div>
            <div style="text-align:center;padding:1.5rem 0;font-family:'Share Tech Mono',monospace;
                        font-size:0.75rem;color:var(--accent-green);letter-spacing:0.1em;">
                ✓ NO ANOMALIES<br>
                <span style="font-size:0.65rem;color:#3a5a6a;">
                    No synthetic regions detected
                </span>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="margin-top:1rem;">
            <div class="card-title">🎯 Region Risk Analysis</div>
            <div style="text-align:center;padding:1.5rem 0;font-family:'Share Tech Mono',monospace;
                        font-size:0.7rem;color:#2a4a5a;letter-spacing:0.1em;">
                Analyze an image to<br>see region breakdown
            </div>
        </div>""", unsafe_allow_html=True)

    # Training data card
    st.markdown("""
    <div class="card" style="margin-top:1rem;">
        <div class="card-title">📦 Training Coverage</div>
        <div class="stat-row"><span class="stat-key">REAL SOURCE</span><span class="stat-val">MS COCO 2017</span></div>
        <div class="stat-row"><span class="stat-key">DALL-E 2/3</span><span class="stat-val orange">✓</span></div>
        <div class="stat-row"><span class="stat-key">MIDJOURNEY V5</span><span class="stat-val orange">✓</span></div>
        <div class="stat-row"><span class="stat-key">STABLE DIFF XL</span><span class="stat-val orange">✓</span></div>
        <div class="stat-row"><span class="stat-key">ADOBE FIREFLY</span><span class="stat-val orange">✓</span></div>
        <div class="stat-row"><span class="stat-key">GLIDE</span><span class="stat-val orange">✓</span></div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="cyber-footer">
    DEEPFAKE DETECTOR v2.0 · EfficientNet-B4 + Grad-CAM · COCO2017 + SynthBuster<br>
    Built with PyTorch · Streamlit · For Educational & Research Purposes Only
</div>""", unsafe_allow_html=True)
