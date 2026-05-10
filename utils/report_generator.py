import io, logging
from datetime import datetime
from PIL import Image as PILImage
import numpy as np
logger = logging.getLogger(__name__)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, Image as RLImage, HRFlowable)
    from reportlab.lib.enums import TA_CENTER
    RL = True
except: RL = False

_SB = {"minor":"#fff3cd","moderate":"#ffe0cc","severe":"#ffd6d6"}

def generate_pdf(image_arr, detections, summary, risk, source="Unknown"):
    if not RL: logger.error("reportlab not installed"); return b""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet(); story = []
    hs = ParagraphStyle("h", fontSize=16, textColor=white, backColor=HexColor("#667eea"),
                        alignment=TA_CENTER, spaceAfter=4, spaceBefore=4, leading=22)
    story.append(Paragraph("🛣️ Road Damage Detection Report — 4-Model Ensemble", hs))
    story.append(Spacer(1,.3*cm))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Source: {source} | "
        f"Risk: {risk.get('level','N/A').upper()} | Models: Unified+PotholeA+PotholeB+RDD",
        ParagraphStyle("s",fontSize=7.5,textColor=HexColor("#555"),alignment=TA_CENTER)))
    story.append(HRFlowable(width="100%",thickness=1,color=HexColor("#667eea")))
    story.append(Spacer(1,.4*cm))
    sv = summary.get("by_severity",{}); tc = risk.get("total_cost",0)
    kd = [["Total","Severe","Moderate","Minor","Risk","Est. Cost (₹)"],
          [str(summary.get("total",len(detections))),str(sv.get("severe",0)),
           str(sv.get("moderate",0)),str(sv.get("minor",0)),
           risk.get("level","N/A").upper(), f"₹{tc:,}"]]
    kt = Table(kd, colWidths=[2.7*cm]*6)
    kt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),HexColor("#667eea")),("TEXTCOLOR",(0,0),(-1,0),white),
        ("FONTSIZE",(0,0),(-1,-1),9),("FONTNAME",(0,1),(-1,1),"Helvetica-Bold"),
        ("FONTSIZE",(0,1),(-1,1),13),("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("GRID",(0,0),(-1,-1),.5,HexColor("#dee2e6")),
        ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
    story.append(kt); story.append(Spacer(1,.4*cm))
    try:
        ib = io.BytesIO(); PILImage.fromarray(image_arr).save(ib,format="JPEG",quality=80); ib.seek(0)
        story.append(RLImage(ib, width=15*cm, height=10*cm, kind="proportional"))
    except Exception as e: logger.warning(e)
    story.append(Spacer(1,.4*cm))
    if detections:
        story.append(Paragraph("Detection Details", styles["Heading2"]))
        dd = [["#","Type","Source","Conf","Severity","Urgency","Action","Cost (₹)"]]
        for i,d in enumerate(detections,1):
            dd.append([str(i), d.get("label","?"), d.get("source","?"),
                       f"{d.get('confidence',0):.1%}", d.get("severity","?").upper(),
                       d.get("urgency","?"),
                       Paragraph(d.get("action","N/A")[:70],
                                 ParagraphStyle("a",fontSize=7,leading=9)),
                       f"₹{d.get('cost_inr',0):,}"])
        dt = Table(dd, colWidths=[.6*cm,2.8*cm,2.2*cm,1.4*cm,1.5*cm,1.5*cm,4.5*cm,1.7*cm])
        ts = [("BACKGROUND",(0,0),(-1,0),HexColor("#764ba2")),("TEXTCOLOR",(0,0),(-1,0),white),
              ("FONTSIZE",(0,0),(-1,-1),7.5),("GRID",(0,0),(-1,-1),.3,HexColor("#dee2e6")),
              ("ALIGN",(0,0),(-1,-1),"CENTER"),("ALIGN",(6,1),(6,-1),"LEFT"),
              ("ROWBACKGROUNDS",(0,1),(-1,-1),[white,HexColor("#f8f9fa")]),
              ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]
        for ri,d in enumerate(detections,1):
            ts.append(("BACKGROUND",(0,ri),(-1,ri),
                       HexColor(_SB.get(d.get("severity","minor"),"#f8f9fa"))))
        dt.setStyle(TableStyle(ts)); story.append(dt)
    story.append(Spacer(1,.5*cm))
    story.append(HRFlowable(width="100%",thickness=.5,color=HexColor("#dee2e6")))
    story.append(Paragraph(
        f"© {datetime.now().year} Srusti (1NT23AD052) — NMIT Bangalore | "
        f"Guides: Prof. Debarshi Mazumder & Prof. Palanivel R",
        ParagraphStyle("f",fontSize=7,textColor=HexColor("#999"),alignment=TA_CENTER)))
    doc.build(story)
    return buf.getvalue()
