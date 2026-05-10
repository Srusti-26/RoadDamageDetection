"""
Hybrid Pipeline v4.1 — 4-Model Ensemble
FIXED: Cracks misclassified as Potholes

Root cause: Pothole-only models (A & B) were firing on crack textures.
Fixes applied:
  1. RDD cls=3 ('Potholes') excluded — noisy label, fires on cracks
  2. SHAPE FILTER: pothole detections must be roughly square (aspect 0.35–2.8)
     and have minimum area. Elongated boxes → likely cracks, not potholes.
  3. CONFLICT RESOLUTION: if a pothole bbox overlaps ≥30% IoU with a crack
     bbox from the RDD/unified model → discard the pothole, keep the crack.
  4. POTHOLE CONF FLOOR raised to 0.45 (crack models stay at base conf).
  5. RDD cls=3 mapped to None → skipped entirely.

Class mapping (output):
  Pothole | Longitudinal Crack | Transverse Crack | Alligator Crack | Crack
"""

import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SKY_THRESHOLD      = 0.25    # ignore boxes whose center_y < 25% frame height
MIN_CONF_CRACK     = 0.35    # crack model minimum confidence
MIN_CONF_POTHOLE   = 0.48    # raised — reduces crack texture false positives
MIN_BOX_AREA       = 400     # px²
NMS_IOU            = 0.45
CROSS_NMS_IOU      = 0.40
FRAME_CONSISTENCY  = 2

# Shape filter for potholes: aspect ratio (w/h) must be in this range
POTHOLE_ASPECT_MIN = 0.30    # not too tall/thin
POTHOLE_ASPECT_MAX = 3.00    # not too wide/flat
POTHOLE_MIN_AREA   = 1200    # potholes are larger than cracks typically

# Conflict resolution: if pothole overlaps crack by this IoU → drop pothole
CRACK_OVERLAP_IOU  = 0.25

# ── Class definitions ─────────────────────────────────────────────────────────
UNIFIED_CLASSES = {0: "Pothole", 1: "Crack"}
# RDD: class 3 = "Potholes" → excluded (noisy, fires on cracks)
RDD_CLASSES     = {0: "Longitudinal Crack", 1: "Transverse Crack", 2: "Alligator Crack"}
# class 3 intentionally omitted → label_fn returns None → skipped

_RISK   = {"Pothole":1.30,"Alligator Crack":1.20,"Transverse Crack":1.05,
           "Longitudinal Crack":1.00,"Crack":1.00}
_THRESH = {"Pothole":(0.42,0.60),"Alligator Crack":(0.43,0.60),
           "Transverse Crack":(0.46,0.62),"Longitudinal Crack":(0.48,0.65),"Crack":(0.44,0.60)}
_ACTIONS = {
    "Pothole":           {"minor":"Cold patch repair","moderate":"Hot-mix asphalt patching","severe":"Full-depth reconstruction"},
    "Alligator Crack":   {"minor":"Fog seal / surface treatment","moderate":"Partial-depth patching","severe":"Full base reconstruction"},
    "Transverse Crack":  {"minor":"Monitor + crack sealant","moderate":"Crack sealing + thin overlay","severe":"Mill and overlay"},
    "Longitudinal Crack":{"minor":"Apply crack sealant","moderate":"Crack filling + surface treat","severe":"Full-depth patching"},
    "Crack":             {"minor":"Monitor + crack sealant","moderate":"Crack filling + treatment","severe":"Mill and overlay"},
}
_URGENCY = {
    "Pothole":           {"minor":"Medium","moderate":"High","severe":"Critical"},
    "Alligator Crack":   {"minor":"Low","moderate":"Medium","severe":"Critical"},
    "Transverse Crack":  {"minor":"Low","moderate":"Medium","severe":"High"},
    "Longitudinal Crack":{"minor":"Low","moderate":"Low","severe":"High"},
    "Crack":             {"minor":"Low","moderate":"Medium","severe":"High"},
}
_COST = {
    "Pothole":           {"minor":3000,"moderate":10000,"severe":30000},
    "Alligator Crack":   {"minor":2000,"moderate":7000, "severe":45000},
    "Transverse Crack":  {"minor":500, "moderate":3500, "severe":14000},
    "Longitudinal Crack":{"minor":400, "moderate":2500, "severe":12000},
    "Crack":             {"minor":500, "moderate":3000, "severe":12000},
}
_COLORS = {
    "Pothole":           (0,50,220),
    "Longitudinal Crack":(255,140,0),
    "Transverse Crack":  (0,200,255),
    "Alligator Crack":   (160,0,255),
    "Crack":             (0,180,100),
}
_SBORDER = {"severe":(30,20,210),"moderate":(0,140,255),"minor":(0,220,100)}


# ── IoU / NMS ─────────────────────────────────────────────────────────────────
def _iou(a, b):
    ix1,iy1 = max(a[0],b[0]),max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]),min(a[3],b[3])
    inter   = max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter == 0: return 0.0
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-6)

def _nms(dets, iou_thr=NMS_IOU):
    if not dets: return []
    boxes  = np.array([d["bbox"] for d in dets], dtype=float)
    scores = np.array([d["confidence"] for d in dets])
    order  = scores.argsort()[::-1]; keep=[]
    while order.size > 0:
        i = order[0]; keep.append(i)
        ious = np.array([_iou(boxes[i],boxes[j]) for j in order[1:]])
        order = order[1:][ious <= iou_thr]
    return [dets[i] for i in keep]


# ── Sky filter ────────────────────────────────────────────────────────────────
def _in_sky(bbox, img_h, thr=SKY_THRESHOLD):
    cy = (bbox[1]+bbox[3])/2.0
    return cy < img_h*thr


# ── Shape filter for potholes ─────────────────────────────────────────────────
def _is_valid_pothole_shape(bbox):
    """
    Potholes are roughly square / blob-shaped.
    Reject if the bounding box is very elongated (likely a crack) or too small.
    """
    bw = max(0, bbox[2]-bbox[0])
    bh = max(0, bbox[3]-bbox[1])
    if bw == 0 or bh == 0: return False
    aspect = bw / bh
    area   = bw * bh
    if area < POTHOLE_MIN_AREA: return False
    if not (POTHOLE_ASPECT_MIN <= aspect <= POTHOLE_ASPECT_MAX): return False
    return True


# ── Conflict resolution ───────────────────────────────────────────────────────
def _remove_pothole_crack_conflicts(pothole_dets, crack_dets, iou_thr=CRACK_OVERLAP_IOU):
    """
    If a pothole detection overlaps significantly with a crack detection,
    discard the pothole — the crack model is more specific.
    """
    if not crack_dets: return pothole_dets
    kept = []
    for pd in pothole_dets:
        conflict = any(_iou(pd["bbox"], cd["bbox"]) >= iou_thr for cd in crack_dets)
        if not conflict:
            kept.append(pd)
        else:
            logger.info(f"Conflict resolution: dropped pothole (overlaps crack) conf={pd['confidence']:.2f}")
    return kept


# ── Inference ─────────────────────────────────────────────────────────────────
def _infer(model, image, conf, label_fn, source):
    dets = []
    try:
        for r in model.predict(image, conf=conf, verbose=False):
            for box in r.boxes.cpu().numpy():
                cls   = int(box.cls.item())
                label = label_fn(cls)
                if label is None: continue
                dets.append({
                    "label": label, "confidence": float(box.conf.item()),
                    "bbox": [int(x) for x in box.xyxy[0].tolist()],
                    "source": source,
                })
    except Exception as e:
        logger.error(f"{source} error: {e}")
    return dets

def _infer_tta(model, image, conf, label_fn, source):
    dets = _infer(model, image, conf, label_fn, source)
    w = image.shape[1]
    flip = cv2.flip(image, 1)
    for d in _infer(model, flip, conf, label_fn, source):
        b = d["bbox"]
        d["bbox"] = [w-b[2], b[1], w-b[0], b[3]]
        dets.append(d)
    return _nms(dets, NMS_IOU)


# ── Severity ──────────────────────────────────────────────────────────────────
def _severity(conf, label, bbox, iw, ih):
    ar = 0.0
    if bbox and len(bbox)==4:
        ar = min((max(0,bbox[2]-bbox[0])*max(0,bbox[3]-bbox[1]))/(iw*ih)*4, 1.0)
    score = min((0.6*conf+0.4*ar)*_RISK.get(label,1.0), 1.0)
    tlo,thi = _THRESH.get(label,(0.46,0.62))
    return ("severe" if score>=thi else "moderate" if score>=tlo else "minor"), round(score,3)


# ── Overall risk ──────────────────────────────────────────────────────────────
def overall_risk(dets):
    if not dets:
        return {"level":"none","score":0.0,"action_required":False,"total_cost":0}
    w  = {"severe":3,"moderate":2,"minor":1}
    tw = sum(w.get(d.get("severity","minor"),1) for d in dets)
    ws = sum(d.get("severity_score",0)*w.get(d.get("severity","minor"),1) for d in dets)/tw
    sc = sum(1 for d in dets if d.get("severity")=="severe")
    mc = sum(1 for d in dets if d.get("severity")=="moderate")
    lvl = "severe" if sc>=1 or ws>=0.62 else "moderate" if mc>=2 or ws>=0.44 else "minor"
    tc  = sum(d.get("cost_inr",0) for d in dets)
    return {"level":lvl,"score":round(ws,3),"action_required":lvl!="minor",
            "severe_count":sc,"moderate_count":mc,"total_cost":tc}


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run(image, unified_model, pothole_a_model, pothole_b_model, rdd_model,
        conf=MIN_CONF_CRACK, use_tta=True, sky_filter=True):
    """
    4-model ensemble with crack/pothole conflict resolution.
    """
    h, w = image.shape[:2]

    # ── 1. Unified model (Pothole=0, Crack=1) ────────────────────────────────
    unified_fn   = lambda c: UNIFIED_CLASSES.get(c)
    unified_dets = (_infer_tta if use_tta else _infer)(
        unified_model, image, conf, unified_fn, "Unified Model")

    # ── 2. Pothole Model A — higher conf threshold ────────────────────────────
    pot_a_fn   = lambda c: "Pothole" if c==0 else None
    pot_a_dets = (_infer_tta if use_tta else _infer)(
        pothole_a_model, image, MIN_CONF_POTHOLE, pot_a_fn, "Pothole Model A")

    # ── 3. Pothole Model B — higher conf threshold ────────────────────────────
    pot_b_fn   = lambda c: "Pothole" if c==0 else None
    pot_b_dets = (_infer_tta if use_tta else _infer)(
        pothole_b_model, image, MIN_CONF_POTHOLE, pot_b_fn, "Pothole Model B")

    # ── 4. RDD crack model — EXCLUDE class 3 (noisy "Potholes" label) ────────
    rdd_fn   = lambda c: RDD_CLASSES.get(c)   # cls=3 → None → skipped
    rdd_dets = _infer(rdd_model, image, conf, rdd_fn, "RDD Crack Model")

    # ── Crack pool (all crack detections) ────────────────────────────────────
    crack_pool = _nms(
        rdd_dets +
        [d for d in unified_dets if d["label"] != "Pothole"],
        NMS_IOU
    )

    # ── Pothole pool ──────────────────────────────────────────────────────────
    # Step 1: collect all pothole candidates
    pot_candidates = (
        pot_a_dets + pot_b_dets +
        [d for d in unified_dets if d["label"] == "Pothole"]
    )

    # Step 2: shape filter — reject elongated/tiny detections
    pot_filtered = [
        d for d in pot_candidates
        if _is_valid_pothole_shape(d["bbox"])
    ]

    # Step 3: conflict resolution — drop potholes overlapping cracks
    pot_filtered = _remove_pothole_crack_conflicts(pot_filtered, crack_pool, CRACK_OVERLAP_IOU)

    # Step 4: NMS within potholes
    pot_pool = _nms(pot_filtered, NMS_IOU)

    # ── Final merge ───────────────────────────────────────────────────────────
    all_dets = _nms(pot_pool + crack_pool, CROSS_NMS_IOU)

    # ── Clean + enrich ────────────────────────────────────────────────────────
    out = []
    for d in all_dets:
        if d["confidence"] < MIN_CONF_CRACK: continue
        b = d["bbox"]
        area = max(0,b[2]-b[0])*max(0,b[3]-b[1]) if len(b)==4 else 0
        if area < MIN_BOX_AREA: continue
        if sky_filter and _in_sky(b, h): continue
        sev, score = _severity(d["confidence"], d["label"], b, w, h)
        d.update({"severity":sev,"severity_score":score,"area_px2":area,
                  "action":  _ACTIONS.get(d["label"],{}).get(sev,"Inspect road"),
                  "urgency": _URGENCY.get(d["label"],{}).get(sev,"Medium"),
                  "cost_inr":_COST.get(d["label"],{}).get(sev,5000)})
        out.append(d)

    out.sort(key=lambda x:{"severe":0,"moderate":1,"minor":2}.get(x["severity"],3))
    logger.info(f"Ensemble: crack={len(crack_pool)} pothole={len(pot_pool)} "
                f"(filtered={len(pot_candidates)-len(pot_filtered)} shape, "
                f"conflict={len(pot_filtered)-len(pot_pool)} iou) → final={len(out)}")
    return out


# ── Frame consistency tracker ─────────────────────────────────────────────────
class FrameConsistencyTracker:
    """Show detection only if seen in min_frames consecutive frames."""
    def __init__(self, min_frames=FRAME_CONSISTENCY, iou_match=0.30):
        self.min_frames = min_frames
        self.iou_match  = iou_match
        self._cands     = {}
        self._key_ctr   = 0

    def _match(self, det):
        best_k, best_iou = None, 0.0
        for k,v in self._cands.items():
            if v["det"]["label"] != det["label"]: continue
            iou = _iou(v["det"]["bbox"], det["bbox"])
            if iou > best_iou: best_iou=iou; best_k=k
        return (best_k, best_iou) if best_iou >= self.iou_match else (None, 0.0)

    def update(self, detections, frame_idx):
        stale = [k for k,v in self._cands.items() if frame_idx-v["last_frame"]>3]
        for k in stale: del self._cands[k]
        for det in detections:
            key,_ = self._match(det)
            if key:
                self._cands[key]["count"]      += 1
                self._cands[key]["last_frame"]  = frame_idx
                self._cands[key]["det"]         = det
            else:
                self._key_ctr += 1
                self._cands[self._key_ctr] = {"det":det,"count":1,"last_frame":frame_idx}
        return [v["det"] for v in self._cands.values()
                if v["count"]>=self.min_frames and frame_idx-v["last_frame"]<=1]

    def reset(self): self._cands.clear(); self._key_ctr=0


# ── Annotation ────────────────────────────────────────────────────────────────
def annotate(image, detections):
    ann = image.copy()
    for d in detections:
        if len(d.get("bbox",[])) != 4: continue
        x1,y1,x2,y2 = d["bbox"]
        sev    = d.get("severity","minor")
        color  = _COLORS.get(d["label"],(100,200,100))
        border = _SBORDER.get(sev,(200,200,200))
        cv2.rectangle(ann,(x1-2,y1-2),(x2+2,y2+2),border,3)
        cv2.rectangle(ann,(x1,y1),(x2,y2),color,2)
        txt = f"{d['label'].split()[0]} {d['confidence']:.0%}[{sev[0].upper()}]"
        (tw,th),_ = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
        cv2.rectangle(ann,(x1,y1-th-6),(x1+tw+4,y1),color,-1)
        cv2.putText(ann,txt,(x1+2,y1-3),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1,cv2.LINE_AA)
    return ann
