"""Video Detection — 4-Model Ensemble | Srusti 1NT23AD052 NMIT"""
import os, cv2, numpy as np, streamlit as st, pandas as pd, plotly.express as px
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from sample_utils.download import download_file
from utils.hybrid_pipeline import run, annotate, overall_risk, FrameConsistencyTracker
from utils.common import apply_css, preprocess, copyright_footer
from utils.history_manager import save_detection
from utils.report_generator import generate_pdf
from utils.email_reporter import send_email

st.set_page_config(page_title="Video Detection", page_icon="🎥", layout="wide")
apply_css()

ROOT = Path(__file__).parent.parent
CRACK_URL  = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"
CRACK_PATH = ROOT/"models/YOLOv8_Small_RDD.pt"
ROOT.joinpath("models").mkdir(exist_ok=True)
download_file(CRACK_URL, CRACK_PATH, expected_size=89569358)

@st.cache_resource
def load_models():
    return (YOLO(ROOT/"models/unified_best.pt"), YOLO(ROOT/"models/best.pt"),
            YOLO(ROOT/"Weights/best.pt"), YOLO(CRACK_PATH))

unified_m, pot_a_m, pot_b_m, rdd_m = load_models()
Path("temp").mkdir(exist_ok=True)
TEMP_IN, TEMP_OUT = "temp/vin.mp4", "temp/vout.mp4"

for k, v in [("vdets",[]),("vstats",{}),("vthumb",None),("vrisk",{})]:
    if k not in st.session_state: st.session_state[k] = v

def process_video(vfile, conf, clahe, fskip, use_tta, sky_filter, frame_consistency):
    with open(TEMP_IN,"wb") as f: f.write(vfile.getbuffer())
    cap = cv2.VideoCapture(TEMP_IN)
    if not cap.isOpened(): st.error("Cannot open video."); return
    W, H   = int(cap.get(3)), int(cap.get(4))
    FPS    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    TOTAL  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    DUR    = TOTAL / FPS

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Duration", f"{int(DUR//60)}:{int(DUR%60):02d}")
    c2.metric("Resolution", f"{W}×{H}")
    c3.metric("FPS", f"{FPS:.1f}")
    c4.metric("Frames", TOTAL)

    writer  = cv2.VideoWriter(TEMP_OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W,H))
    pbar    = st.progress(0, "Initializing…")
    pc, sc  = st.columns([2,1])
    prev_ph = pc.empty(); live_ph = sc.empty()
    tracker = FrameConsistencyTracker(min_frames=frame_consistency)
    all_dets= []; fn = 0; thumb = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if fn % fskip != 0:
            writer.write(frame); fn += 1; continue

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        proc = preprocess(rgb, (640,640), enhance=clahe)

        raw_dets = run(proc, unified_m, pot_a_m, pot_b_m, rdd_m,
                       conf=conf, use_tta=use_tta, sky_filter=sky_filter)

        stable = tracker.update(raw_dets, fn) if frame_consistency > 1 else raw_dets

        for d in stable:
            d["frame"] = fn; d["timestamp"] = round(fn/FPS, 2)
        all_dets.extend(stable)

        ann = cv2.cvtColor(cv2.resize(annotate(proc, stable),(W,H)), cv2.COLOR_RGB2BGR)
        writer.write(ann)

        if fn % (fskip*8) == 0:
            thumb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
            prev_ph.image(thumb, use_container_width=True)

        np2 = sum(1 for d in all_dets if d["label"]=="Pothole")
        live_ph.markdown(f"**Frame:** {fn}/{TOTAL}\n\n**Total:** {len(all_dets)}\n\n"
                         f"⚠️ Potholes: {np2}\n\n🔍 Cracks: {len(all_dets)-np2}")
        fn += 1
        pbar.progress(min(fn/max(TOTAL,1), 1.0), text=f"Frame {fn}/{TOTAL}…")

    cap.release(); writer.release(); pbar.empty()

    # Temporal dedup
    seen = {}
    deduped = []
    for d in all_dets:
        k = (d["label"], d.get("frame",0)//5)
        if k not in seen or d["confidence"] > seen[k]["confidence"]:
            seen[k] = d
    deduped = list(seen.values())

    save_detection(deduped, "video_upload")
    risk = overall_risk(deduped)
    np2  = sum(1 for d in deduped if d["label"]=="Pothole")
    st.session_state.vdets  = deduped
    st.session_state.vrisk  = risk
    st.session_state.vstats = {"total_frames":TOTAL,"duration":DUR,"fps":FPS}
    st.session_state.vthumb = thumb

    rc = {"severe":"🔴","moderate":"🟠","minor":"🟡"}.get(risk["level"],"⚪")
    st.success(f"✅ {len(deduped)} unique detections — {len(deduped)-np2} crack(s) + {np2} pothole(s) "
               f"| {rc} Risk: {risk['level'].upper()} | Est: ₹{risk['total_cost']:,}")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Video Settings")
    conf             = st.slider("Confidence",0.25,0.85,0.38,0.05)
    fskip            = st.slider("Frame Skip",1,15,3)
    frame_consistency= st.slider("Frame Consistency",1,5,2,help="Detect only if seen in N frames")
    use_tta          = st.toggle("TTA",True)
    clahe            = st.toggle("CLAHE",True)
    sky_filter       = st.toggle("Sky Filter",True)
    st.markdown("---\n### Models")
    st.success("✅ Unified"); st.success("✅ Pothole A")
    st.success("✅ Pothole B"); st.success("✅ RDD Crack")
    email_en = st.toggle("📧 Email",False)

st.markdown("""<div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:1.5rem;
border-radius:12px;color:white;text-align:center;margin-bottom:1rem">
<h2 style="margin:0">🎥 Hybrid Video Detection — 4-Model Ensemble</h2>
<p style="margin:4px 0 0;opacity:.85">Frame Consistency Filter · Sky Filter · Temporal Dedup · TTA</p>
</div>""",unsafe_allow_html=True)

vf = st.file_uploader("Upload road video",type=["mp4","avi","mov"])
if vf and st.button("🚀 Process Video",type="primary",use_container_width=True):
    process_video(vf, conf, clahe, fskip, use_tta, sky_filter, frame_consistency)

t1,t2,t3 = st.tabs(["📹 Processed Video","📊 Analysis","📄 Reports"])

with t1:
    if os.path.exists(TEMP_OUT) and st.session_state.vdets:
        try:
            with open(TEMP_OUT,"rb") as f: st.video(f.read())
        except: st.info("Video ready. Re-run if preview fails.")
        with open(TEMP_OUT,"rb") as f:
            st.download_button("📥 Download Video",f.read(),
                f"RDD_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4","video/mp4",use_container_width=True)
    else: st.info("Upload and process a video.")

with t2:
    dets = st.session_state.vdets; stats = st.session_state.vstats; risk = st.session_state.vrisk
    if dets:
        np2 = sum(1 for d in dets if d["label"]=="Pothole")
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Total",len(dets)); k2.metric("Cracks",len(dets)-np2)
        k3.metric("Potholes",np2); k4.metric("Duration",f"{stats.get('duration',0):.1f}s")
        k5.metric("Est. Cost",f"₹{risk.get('total_cost',0):,}")

        rc = {"severe":"#DC143C","moderate":"#FF6B6B","minor":"#FFA500"}.get(risk.get("level","minor"),"#aaa")
        st.markdown(f"""<div style="background:{rc};color:white;padding:8px;border-radius:8px;
        text-align:center;font-weight:700;margin:.5rem 0">
        🚨 Overall Risk: {risk.get('level','N/A').upper()} (Score: {risk.get('score',0):.2f})
        </div>""",unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        bt = {}
        for d in dets: bt[d["label"]] = bt.get(d["label"],0)+1
        with c1:
            fig = px.pie(values=list(bt.values()),names=list(bt.keys()),
                         title="Damage Distribution",hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=290,margin=dict(t=50,b=10,l=10,r=10))
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            df_t = pd.DataFrame([{"time":d.get("timestamp",0),"type":d["label"],
                                   "conf":d["confidence"]} for d in dets])
            fig2 = px.scatter(df_t,x="time",y="conf",color="type",title="Detection Timeline",
                              labels={"time":"Time (s)","conf":"Confidence"})
            fig2.update_layout(height=290,margin=dict(t=50,b=10,l=10,r=10))
            st.plotly_chart(fig2,use_container_width=True)

        # Density heatmap
        if len(dets) > 1:
            try:
                df_t["second"] = df_t["time"].astype(int)
                hm = df_t.groupby(["second","type"]).size().unstack(fill_value=0)
                if not hm.empty:
                    fig3 = px.imshow(hm.T.values,x=[f"{s}s" for s in hm.index],
                                     y=hm.columns.tolist(),color_continuous_scale="YlOrRd",
                                     title="Detection Density Heatmap")
                    fig3.update_layout(height=240,margin=dict(t=50,b=10,l=10,r=10))
                    st.plotly_chart(fig3,use_container_width=True)
            except: pass

        st.markdown("#### 📋 Detection Log")
        rows = [{"Frame":d.get("frame",""),"Time(s)":d.get("timestamp",""),
                 "Type":d["label"],"Model":d["source"],"Conf":f"{d['confidence']:.1%}",
                 "Severity":d.get("severity","").upper(),"Action":d.get("action","")}
                for d in dets[:300]]
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    else: st.info("Process a video first.")

with t3:
    dets = st.session_state.vdets; thumb = st.session_state.vthumb; risk = st.session_state.vrisk
    if dets:
        sv = {"minor":sum(1 for d in dets if d.get("severity")=="minor"),
              "moderate":sum(1 for d in dets if d.get("severity")=="moderate"),
              "severe":sum(1 for d in dets if d.get("severity")=="severe")}
        summary = {"total":len(dets),"by_severity":sv}
        r1,r2 = st.columns(2)
        with r1:
            if st.button("📕 Generate PDF",use_container_width=True):
                img = thumb if thumb is not None else np.zeros((100,100,3),dtype=np.uint8)
                pdf = generate_pdf(img,dets[:60],summary,risk,"Video Upload")
                if pdf: st.download_button("📥 Download PDF",pdf,
                    f"VideoReport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    "application/pdf",use_container_width=True)
                else: st.error("Install: pip install reportlab")
        with r2:
            if st.button("📗 Generate CSV",use_container_width=True):
                csv = pd.DataFrame([{"Type":d["label"],"Model":d["source"],
                    "Conf":f"{d['confidence']:.1%}","Severity":d.get("severity",""),
                    "Action":d.get("action",""),"Cost INR":d.get("cost_inr",0)}
                    for d in dets]).to_csv(index=False)
                st.download_button("📥 Download CSV",csv,
                    f"VideoReport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",use_container_width=True)

        if email_en:
            st.divider(); st.markdown("#### 📧 Email Report")
            st.caption("Requires Gmail App Password — [generate here](https://myaccount.google.com/apppasswords)")
            e1,e2 = st.columns(2)
            with e1:
                se = st.text_input("Sender Gmail",key="vse")
                sp = st.text_input("App Password",type="password",key="vsp")
            with e2:
                re  = st.text_input("Recipient",key="vre")
                att = st.checkbox("Attach PDF",True,key="vatt")
            if st.button("📤 Send",type="primary",use_container_width=True):
                if not all([se,sp,re]): st.error("Fill all fields.")
                else:
                    pdf_a = None
                    if att:
                        img = thumb if thumb is not None else np.zeros((100,100,3),dtype=np.uint8)
                        pdf_a = generate_pdf(img,dets[:60],summary,risk,"Video Upload")
                    with st.spinner("Sending…"):
                        ok,msg = send_email(se,sp,re,dets[:60],summary,risk,"Video Detection",pdf_a)
                    (st.success if ok else st.error)(msg)
    else: st.info("Process a video first.")

copyright_footer()
