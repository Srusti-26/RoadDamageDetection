"""Home | Srusti 1NT23AD052 NMIT Bangalore"""
import streamlit as st, plotly.express as px, plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Road Damage Detection",page_icon="🛣️",layout="wide",initial_sidebar_state="expanded")
st.markdown("""<style>
.main-hdr{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);padding:2rem;border-radius:14px;color:white;text-align:center;margin-bottom:1rem;}
.guide-card{background:linear-gradient(135deg,#667eea,#764ba2);padding:1.2rem;border-radius:10px;color:white;text-align:center;}
.model-card{background:#f8f9ff;border:2px solid #667eea;border-radius:10px;padding:.9rem;text-align:center;}
.stButton>button{background:linear-gradient(135deg,#667eea,#764ba2)!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e,#16213e)!important;}
[data-testid="stSidebar"] *{color:white!important;}
div[data-testid="metric-container"]{background:#f8f9ff;border-radius:10px;padding:10px;border:1px solid #e0e4ff;}
</style>""",unsafe_allow_html=True)

try:
    from utils.history_manager import get_history_stats
    h = get_history_stats()
except: h = {"total":0,"by_type":{},"by_severity":{"minor":0,"moderate":0,"severe":0}}

st.markdown("""<div class="main-hdr">
  <div style="font-size:3rem">🛣️</div>
  <h1 style="margin:0;font-size:1.7rem;letter-spacing:.5px">Hybrid Deep Learning-Based Smart Road Damage Detection System</h1>
  <p style="margin:.6rem 0 .3rem;opacity:.85;font-size:.95rem">4-Model Ensemble · TTA · CLAHE · Sky Filter · Frame Consistency · AI Severity Analysis</p>
  <hr style="border-color:rgba(255,255,255,.2);margin:.8rem 0">
  <div style="display:flex;justify-content:center;gap:3rem;flex-wrap:wrap">
    <div><div style="font-size:.7rem;opacity:.7;text-transform:uppercase;letter-spacing:1px">Student</div><b>Srusti</b></div>
    <div><div style="font-size:.7rem;opacity:.7;text-transform:uppercase;letter-spacing:1px">USN</div><b>1NT23AD052</b></div>
    <div><div style="font-size:.7rem;opacity:.7;text-transform:uppercase;letter-spacing:1px">College</div><b>NMIT Bangalore</b></div>
  </div>
</div>""",unsafe_allow_html=True)

st.markdown("### 👨‍🏫 Project Guides")
g1,g2=st.columns(2)
with g1: st.markdown("""<div class="guide-card"><div style="font-size:2rem">👨‍💼</div>
    <h3 style="margin:.4rem 0 .2rem">Prof. Debarshi Mazumder</h3><p style="opacity:.85;margin:0;font-size:.9rem">Project Guide</p>
    <div style="margin-top:.6rem"><span style="background:rgba(255,255,255,.2);padding:3px 10px;border-radius:14px;font-size:.8rem;margin:2px">ANN</span>
    <span style="background:rgba(255,255,255,.2);padding:3px 10px;border-radius:14px;font-size:.8rem;margin:2px">Deep Learning</span></div></div>""",unsafe_allow_html=True)
with g2: st.markdown("""<div class="guide-card"><div style="font-size:2rem">👨‍💼</div>
    <h3 style="margin:.4rem 0 .2rem">Prof. Palanivel R</h3><p style="opacity:.85;margin:0;font-size:.9rem">Project Guide</p>
    <div style="margin-top:.6rem"><span style="background:rgba(255,255,255,.2);padding:3px 10px;border-radius:14px;font-size:.8rem;margin:2px">DIP</span>
    <span style="background:rgba(255,255,255,.2);padding:3px 10px;border-radius:14px;font-size:.8rem;margin:2px">Computer Vision</span></div></div>""",unsafe_allow_html=True)

st.markdown("---")
for col,(icon,short,full) in zip(st.columns(4),[("🧠","ANN","Artificial Neural Networks"),("⚡","DL","Deep Learning"),
                                                  ("🖼️","DIP","Digital Image Processing"),("👁️","CV","Computer Vision")]):
    col.markdown(f"""<div style="background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:.9rem;border-radius:8px;text-align:center">
    <div style="font-size:1.4rem">{icon}</div><b>{short}</b><div style="font-size:.75rem;opacity:.85">{full}</div></div>""",unsafe_allow_html=True)

st.markdown("---\n### 🔬 Model Architecture")
cols = st.columns(3)
for col, (icon, title, sub, clr) in zip(cols,[
    ("👁️","YOLOv8n","Object Detection\nBounding boxes + class labels\nCOCO pretrained → RDD2022 FT\n~3.2M params · Input: 640×640","#4facfe"),
    ("⚡","MobileNetV2","Transfer Learning Classifier\nImageNet backbone (frozen)\nCustom head: 1280→256→5\n~329K trainable params","#667eea"),
    ("🧠","Custom CNN","From-Scratch Classifier\n4 Conv blocks: 32→64→128→256\nAdaptiveAvgPool → FC(512→5)\n~2.5M params · RDD2022 trained","#764ba2")]):
    col.markdown(f"""<div style="background:#f8f9ff;border:2px solid {clr};border-radius:10px;padding:1rem;text-align:center">
    <div style="font-size:1.8rem">{icon}</div><h5 style="color:{clr};margin:.4rem 0">{title}</h5>
    <div style="font-size:.75rem;white-space:pre-line;color:#555;line-height:1.6">{sub}</div></div>""",unsafe_allow_html=True)

st.markdown("""<div style="background:#f0f4ff;border-radius:10px;padding:.9rem;text-align:center;font-weight:600;color:#667eea;margin-top:.5rem">
Input Image/Video → Preprocess → YOLOv8n Detection + MobileNetV2 &amp; Custom CNN Classification → Severity Analysis → Output
</div>""",unsafe_allow_html=True)

st.markdown("---\n### 📊 Session Statistics")
k1,k2,k3,k4,k5=st.columns(5)
for col,val,label,grad in [
    (k1,h["total"],"Total","linear-gradient(135deg,#667eea,#764ba2)"),
    (k2,h["by_severity"].get("severe",0),"🔴 Severe","linear-gradient(135deg,#DC143C,#ff4444)"),
    (k3,h["by_severity"].get("moderate",0),"🟠 Moderate","linear-gradient(135deg,#FF6B6B,#ffa500)"),
    (k4,h["by_severity"].get("minor",0),"🟡 Minor","linear-gradient(135deg,#FFA500,#ffd700)"),
    (k5,h["by_type"].get("Pothole",0),"⚠️ Potholes","linear-gradient(135deg,#4facfe,#00f2fe)")]:
    col.markdown(f"""<div style="background:{grad};color:white;padding:1rem;border-radius:8px;text-align:center">
    <h2 style="margin:0">{val}</h2><small>{label}</small></div>""",unsafe_allow_html=True)

if h["total"]>0 and h["by_type"]:
    c1,c2=st.columns(2)
    with c1:
        fig=px.pie(values=list(h["by_type"].values()),names=list(h["by_type"].keys()),
                   title="Damage Distribution",hole=0.4,color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=270,margin=dict(t=50,b=10,l=10,r=10)); st.plotly_chart(fig,use_container_width=True)
    with c2:
        sv=h["by_severity"]
        fig2=px.bar(x=["Minor","Moderate","Severe"],y=[sv.get("minor",0),sv.get("moderate",0),sv.get("severe",0)],
                    title="Severity Breakdown",color=["Minor","Moderate","Severe"],
                    color_discrete_map={"Minor":"#FFA500","Moderate":"#FF6B6B","Severe":"#DC143C"},text_auto=True)
        fig2.update_layout(showlegend=False,height=270,margin=dict(t=50,b=10,l=10,r=10)); st.plotly_chart(fig2,use_container_width=True)

st.markdown("---\n### 🚀 Quick Navigation")
n1,n2,n3,n4=st.columns(4)
with n1:
    if st.button("🖼️ Image Detection",use_container_width=True,type="primary"): st.switch_page("pages/2_Image_Detection.py")
with n2:
    if st.button("🎥 Video Detection",use_container_width=True): st.switch_page("pages/3_Video_Detection.py")
with n3:
    if st.button("📊 Dashboard",use_container_width=True): st.switch_page("pages/4_Dashboard.py")
with n4:
    if st.button("🕐 History",use_container_width=True): st.switch_page("pages/5_History.py")

st.markdown(f"""<hr><div style="text-align:center;background:#f8f9ff;padding:.8rem;border-radius:8px;color:#667eea;font-size:.85rem;font-weight:600">
© {datetime.now().year} Srusti (1NT23AD052) — NMIT Bangalore | Guides: Prof. Debarshi Mazumder & Prof. Palanivel R | ANN · DL · DIP · CV
</div>""",unsafe_allow_html=True)
