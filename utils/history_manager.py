import csv, os
from datetime import datetime
from pathlib import Path

HISTORY_FILE = Path("reports/history.csv")
COLS = ["timestamp","source","damage_type","confidence","severity",
        "bbox_x1","bbox_y1","bbox_x2","bbox_y2","area","model","action"]

def _ensure():
    HISTORY_FILE.parent.mkdir(exist_ok=True)
    if not HISTORY_FILE.exists():
        with open(HISTORY_FILE,"w",newline="") as f:
            csv.DictWriter(f, fieldnames=COLS).writeheader()

def save_detection(dets, source="unknown"):
    _ensure()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for d in dets:
        b = d.get("bbox",[0,0,0,0])
        a = (b[2]-b[0])*(b[3]-b[1]) if len(b)==4 else 0
        rows.append({"timestamp":ts,"source":source,"damage_type":d.get("label","Unknown"),
            "confidence":round(d.get("confidence",0),4),"severity":d.get("severity","minor"),
            "bbox_x1":b[0]if len(b)>0 else 0,"bbox_y1":b[1]if len(b)>1 else 0,
            "bbox_x2":b[2]if len(b)>2 else 0,"bbox_y2":b[3]if len(b)>3 else 0,
            "area":a,"model":d.get("source","ensemble"),"action":d.get("action","")})
    with open(HISTORY_FILE,"a",newline="") as f:
        csv.DictWriter(f, fieldnames=COLS).writerows(rows)

def get_history_stats():
    _ensure()
    recs = list(csv.DictReader(open(HISTORY_FILE)))
    if not recs: return {"total":0,"by_type":{},"by_severity":{"minor":0,"moderate":0,"severe":0}}
    bt, bs = {}, {"minor":0,"moderate":0,"severe":0}
    for r in recs:
        dt=r.get("damage_type","Unknown"); sv=r.get("severity","minor")
        bt[dt] = bt.get(dt,0)+1
        if sv in bs: bs[sv] += 1
    return {"total":len(recs),"by_type":bt,"by_severity":bs}

def load_history_df():
    import pandas as pd; _ensure()
    try: return pd.read_csv(HISTORY_FILE)
    except: return pd.DataFrame(columns=COLS)

def clear_history():
    if HISTORY_FILE.exists(): os.remove(HISTORY_FILE)
    _ensure()
