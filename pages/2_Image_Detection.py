"""Image Detection — 4-Model Ensemble | Srusti 1NT23AD052 NMIT"""
import streamlit as st, cv2, numpy as np, pandas as pd
from PIL import Image
from io import BytesIO
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from sample_utils.download import download_file
from utils.hybrid_pipeline import run, annotate, overall_risk
from utils.common import apply_css, preprocess, init_state, copyright_footer
from utils.history_manager import save_detection
from utils.email_reporter import send_email
from utils.report_generator import generate_pdf

st.set_page_config(page_title="Image Detection",page_icon="🖼️",layout="wide")
apply_css(); init_state()

ROOT = Path(__file__).parent.parent
CRACK_URL  = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"
CRACK_PATH = ROOT/"models/YOLOv8_Small_RDD.pt"
ROOT.joinpath("models").mkdir(exist_ok=True)
download_file(CRACK_URL, CRACK_PATH, expected_size=89569358)

@st.cache_resource
def load_models():
    unified  = YOLO(ROOT/"models/unified_best.pt")
    pot_a    = YOLO(ROOT/"models/best.pt")
    pot_b    = YOLO(ROOT/"Weights/best.pt")
    rdd      = YOLO(CRACK_PATH)
    return unified, pot_a, pot_b, rdd

unified_m, pot_a_m, pot_b_m, rdd_m = load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    conf      = st.slider("Confidence Threshold",0.25,0.85,0.38,0.05)
    use_tta   = st.toggle("TTA (Test-Time Aug.)",True,help="Horizontal flip augmentation")
    use_clahe = st.toggle("CLAHE Enhancement",True)
    sky_filter= st.toggle("Sky Filter",True,help="Ignore detections in top 25% of frame")
    st.markdown("---\n### 🤖 Models Loaded")
    st.success("✅ Unified (Pothole+Crack)")
    st.success("✅ Pothole Model A (DS_A)")
    st.success("✅ Pothole Model B (DS_B)")
    st.success("✅ RDD Crack Model")
    email_en  = st.toggle("📧 Email Report",False)

st.markdown("""<div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:1.5rem;
border-radius:12px;color:white;text-align:center;margin-bottom:1rem">
<h2 style="margin:0">🖼️ Hybrid Image Detection — 4-Model Ensemble</h2>
<p style="margin:4px 0 0;opacity:.85">Unified + Pothole A + Pothole B + RDD Crack → NMS Merge → Sky Filter → Severity</p>
</div>""",unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Detection & Results","📄 Export Reports","📧 Email Report"])

with tab1:
    c1,c2=st.columns(2)
    with c1:
        st.markdown("#### 📤 Upload Image")
        f = st.file_uploader("Road image (JPG/PNG)",type=["png","jpg","jpeg"])
        if f:
            try:
                pil = Image.open(BytesIO(f.read())).convert("RGB")
            except Exception as e:
                st.error(f"❌ Could not open image: {e}")
                st.stop()
            st.image(pil,"Original",use_container_width=True)
            if st.button("🚀 Run 4-Model Ensemble",type="primary",use_container_width=True):
                with st.spinner("🔍 Running all 4 models…"):
                    arr = np.array(pil); h_o,w_o = arr.shape[:2]
                    proc = preprocess(arr,(640,640),enhance=use_clahe)
                    dets = run(proc, unified_m, pot_a_m, pot_b_m, rdd_m,
                               conf=conf, use_tta=use_tta, sky_filter=sky_filter)
                    ann  = cv2.resize(annotate(proc,dets),(w_o,h_o))
                    st.session_state.detections = dets
                    st.session_state.annotated  = ann
                    save_detection(dets,"image_upload")

                risk = overall_risk(dets)
                n_p  = sum(1 for d in dets if d["label"]=="Pothole")
                rc   = {"severe":"🔴","moderate":"🟠","minor":"🟡"}.get(risk["level"],"⚪")
                st.success(f"✅ {len(dets)} detections — {n_p} pothole(s) + {len(dets)-n_p} crack(s) "
                           f"| {rc} Risk: {risk['level'].upper()} | ₹{risk['total_cost']:,}")

    with c2:
        st.markdown("#### 🎯 Results")
        dets = st.session_state.get("detections")
        ann  = st.session_state.get("annotated")
        if dets is not None and ann is not None:
            st.image(ann,"Detection Result",use_container_width=True)
            buf=BytesIO(); Image.fromarray(ann).save(buf,format="PNG")
            st.download_button("📥 Download",buf.getvalue(),
                f"RDD_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png","image/png",use_container_width=True)

            if dets:
                risk=overall_risk(dets)
                rc={"severe":"#DC143C","moderate":"#FF6B6B","minor":"#FFA500"}.get(risk["level"],"#aaa")
                st.markdown(f"""<div style="background:{rc};color:white;padding:9px;border-radius:8px;
                text-align:center;font-weight:700;margin:.5rem 0">
                🚨 Risk: {risk['level'].upper()} | Score: {risk['score']:.2f} | Est: ₹{risk['total_cost']:,}
                </div>""",unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("#### 📊 Detection Cards")
                _SB={"minor":"#fff3cd","moderate":"#ffe0cc","severe":"#ffd6d6"}
                _ST={"minor":"#856404","moderate":"#7d3c00","severe":"#7b0000"}
                _UI={"Critical":"🔴","High":"🟠","Medium":"🟡","Low":"🟢"}
                _DI={"Pothole":"⚠️","Longitudinal Crack":"〰️","Transverse Crack":"➖",
                     "Alligator Crack":"🕸️","Crack":"〰️"}
                for i,d in enumerate(dets,1):
                    sev=d.get("severity","minor"); bg=_SB.get(sev,"#f8f9fa"); tc=_ST.get(sev,"#333")
                    pct=int(d.get("confidence",0)*100)
                    bc="#28a745" if pct>=75 else "#fd7e14" if pct>=55 else "#dc3545"
                    st.markdown(f"""<div style="background:{bg};border-left:5px solid {tc};border-radius:10px;padding:13px;margin-bottom:9px">
                        <div style="display:flex;justify-content:space-between;align-items:center">
                            <span style="font-weight:700;color:#1a1a2e">{_DI.get(d['label'],'🔍')} #{i} {d['label']}</span>
                            <span style="background:{tc};color:white;padding:2px 9px;border-radius:16px;font-size:.72rem;font-weight:700">{sev.upper()}</span>
                        </div>
                        <div style="background:#e9ecef;border-radius:5px;height:9px;width:100%;margin:7px 0 3px">
                            <div style="width:{pct}%;background:{bc};height:9px;border-radius:5px"></div>
                        </div>
                        <small style="color:{bc};font-weight:600">{pct}% confidence</small>
                        <div style="display:flex;gap:12px;margin-top:7px;flex-wrap:wrap">
                            <small>📦 <b>Model:</b> {d.get('source','')}</small>
                            <small>📐 <b>Area:</b> {d.get('area_px2',0):,} px²</small>
                            <small>⚡ {_UI.get(d.get('urgency','Medium'),'🟡')} {d.get('urgency','')}</small>
                            <small>💰 ₹{d.get('cost_inr',0):,}</small>
                        </div>
                        <div style="background:white;border-radius:5px;padding:7px;margin-top:7px">
                            <small>🔧 <b>Action:</b> {d.get('action','Inspect road')}</small>
                        </div>
                    </div>""",unsafe_allow_html=True)
        elif dets is not None and not dets:
            st.info("No damage detected. Try lowering confidence threshold.")
        else:
            st.info("👆 Upload an image and click 'Run 4-Model Ensemble'")

with tab2:
    dets = st.session_state.get("detections"); ann = st.session_state.get("annotated")
    if dets is not None:
        risk = overall_risk(dets)
        sv   = {"minor":sum(1 for d in dets if d.get("severity")=="minor"),
                "moderate":sum(1 for d in dets if d.get("severity")=="moderate"),
                "severe":sum(1 for d in dets if d.get("severity")=="severe")}
        summary = {"total":len(dets),"by_severity":sv}
        r1,r2   = st.columns(2)
        with r1:
            if st.button("📕 Generate PDF",use_container_width=True):
                img = ann if ann is not None else np.zeros((100,100,3),dtype=np.uint8)
                pdf = generate_pdf(img,dets,summary,risk,"Image Upload")
                if pdf: st.download_button("📥 Download PDF",pdf,
                    f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf","application/pdf",use_container_width=True)
                else: st.error("Install: pip install reportlab")
        with r2:
            if st.button("📗 Generate CSV",use_container_width=True):
                csv=pd.DataFrame([{"Damage":d["label"],"Model":d["source"],
                    "Confidence":f"{d['confidence']:.1%}","Severity":d["severity"],
                    "Urgency":d.get("urgency",""),"Action":d.get("action",""),
                    "Cost INR":d.get("cost_inr",0)} for d in dets]).to_csv(index=False)
                st.download_button("📥 Download CSV",csv,
                    f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv","text/csv",use_container_width=True)
        if dets:
            st.markdown("---\n#### 📋 Damage Summary")
            st.dataframe(pd.DataFrame([{"#":i,"Type":d["label"],"Model":d["source"],
                "Confidence":f"{d['confidence']:.1%}","Severity":d["severity"].upper(),
                "Urgency":d.get("urgency",""),"Action":d.get("action",""),
                "Cost ₹":d.get("cost_inr",0)} for i,d in enumerate(dets,1)]),
                use_container_width=True, hide_index=True)
    else: st.info("Run a detection first.")

with tab3:
    if not email_en: st.info("Enable '📧 Email Report' in sidebar.")
    else:
        dets = st.session_state.get("detections"); ann = st.session_state.get("annotated")
        if not dets: st.info("Run a detection first.")
        else:
            st.markdown("#### 📧 Send via Gmail")
            st.caption("Requires Gmail App Password — [generate here](https://myaccount.google.com/apppasswords)")
            e1,e2 = st.columns(2)
            with e1:
                se = st.text_input("Sender Gmail",placeholder="you@gmail.com")
                sp = st.text_input("App Password",type="password",placeholder="xxxx xxxx xxxx xxxx")
            with e2:
                re = st.text_input("Recipient",placeholder="engineer@example.com")
                att= st.checkbox("Attach PDF",True)
            if st.button("📤 Send Report",type="primary",use_container_width=True):
                if not all([se,sp,re]): st.error("Fill all fields.")
                else:
                    risk=overall_risk(dets)
                    sv={"minor":sum(1 for d in dets if d.get("severity")=="minor"),
                        "moderate":sum(1 for d in dets if d.get("severity")=="moderate"),
                        "severe":sum(1 for d in dets if d.get("severity")=="severe")}
                    summary={"total":len(dets),"by_severity":sv}
                    pdf_att=None
                    if att:
                        img=ann if ann is not None else np.zeros((100,100,3),dtype=np.uint8)
                        pdf_att=generate_pdf(img,dets,summary,risk,"Image Upload")
                    with st.spinner("Sending…"):
                        ok,msg=send_email(se,sp,re,dets,summary,risk,"Image Detection",pdf_att)
                    (st.success if ok else st.error)(msg)

copyright_footer()
