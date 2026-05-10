import cv2, numpy as np, streamlit as st, logging
from pathlib import Path
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.WARNING,
    handlers=[logging.FileHandler("logs/app.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

CSS = """<style>
.stButton>button{background:linear-gradient(135deg,#667eea,#764ba2)!important;color:white!important;
  border:none!important;border-radius:8px!important;font-weight:600!important;transition:all .2s!important;}
.stButton>button:hover{opacity:.9!important;transform:translateY(-1px)!important;}
div[data-testid="metric-container"]{background:#f8f9ff;border-radius:10px;padding:10px;border:1px solid #e0e4ff;}
.stTabs [data-baseweb="tab"]{font-weight:600;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e,#16213e)!important;}
[data-testid="stSidebar"] *{color:white!important;}
</style>"""

def apply_css(): st.markdown(CSS, unsafe_allow_html=True)

def init_state():
    for k, v in [("detections",None),("annotated",None),
                 ("vdets",[]),("vthumb",None),("vstats",{}),("vrisk",{})]:
        if k not in st.session_state: st.session_state[k] = v

def preprocess(img, size=(640,640), enhance=True):
    try:
        r = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
        if enhance:
            lab = cv2.cvtColor(r, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clip = 3.5 if np.mean(l) < 110 else 2.0
            l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8)).apply(l)
            r = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
            r = cv2.bilateralFilter(r, 5, 50, 50)
        return r
    except: return img

def copyright_footer():
    from datetime import datetime
    st.markdown(f"""<hr style="margin:1.5rem 0 .5rem">
    <div style="text-align:center;color:#888;font-size:.8rem">
    © {datetime.now().year} <b>Srusti</b> (1NT23AD052) — NMIT Bangalore |
    ANN · DL · DIP · CV
    </div>""", unsafe_allow_html=True)
