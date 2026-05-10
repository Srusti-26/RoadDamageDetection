import smtplib, ssl, logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
logger = logging.getLogger(__name__)

def _html(dets, summary, risk, source):
    sv = summary.get("by_severity",{}); tc = risk.get("total_cost",0)
    rl = risk.get("level","minor")
    sbc={"minor":"#fff3cd","moderate":"#ffe0cc","severe":"#ffd6d6"}
    stc={"minor":"#856404","moderate":"#7d3c00","severe":"#7b0000"}
    rows = "".join(f"""<tr style="background:{sbc.get(d.get('severity','minor'),'#f8f9fa')}">
    <td style="padding:7px;text-align:center">{i}</td>
    <td style="padding:7px">{d.get('label','?')}</td>
    <td style="padding:7px;font-size:11px">{d.get('source','?')}</td>
    <td style="padding:7px;text-align:center">{d.get('confidence',0):.1%}</td>
    <td style="padding:7px;text-align:center;color:{stc.get(d.get('severity','minor'),'#333')};font-weight:700">{d.get('severity','?').upper()}</td>
    <td style="padding:7px;font-size:11px">{d.get('action','N/A')}</td>
    <td style="padding:7px;text-align:right">₹{d.get('cost_inr',0):,}</td>
    </tr>""" for i,d in enumerate(dets,1))
    return f"""<!DOCTYPE html><html><body style="font-family:Arial,sans-serif;background:#f4f6f9;margin:0">
<div style="max-width:720px;margin:20px auto;background:white;border-radius:12px;overflow:hidden;box-shadow:0 4px 12px rgba(0,0,0,.1)">
  <div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:20px;text-align:center;color:white">
    <h2 style="margin:0">🛣️ Road Damage Report — 4-Model Ensemble</h2>
    <p style="margin:4px 0 0;opacity:.85;font-size:12px">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · {source}</p>
    <p style="margin:2px 0 0;opacity:.7;font-size:11px">Srusti (1NT23AD052) — NMIT Bangalore</p>
  </div>
  <div style="padding:20px">
    <table style="width:100%;border-collapse:collapse;margin-bottom:16px"><tr>
      <td style="padding:12px;text-align:center;background:#667eea;color:white;border-radius:8px;width:16%"><b style="font-size:20px">{summary.get('total',len(dets))}</b><br><small>Total</small></td>
      <td style="width:2%"></td>
      <td style="padding:12px;text-align:center;background:#DC143C;color:white;border-radius:8px;width:16%"><b style="font-size:20px">{sv.get('severe',0)}</b><br><small>Severe</small></td>
      <td style="width:2%"></td>
      <td style="padding:12px;text-align:center;background:#FF6B6B;color:white;border-radius:8px;width:16%"><b style="font-size:20px">{sv.get('moderate',0)}</b><br><small>Moderate</small></td>
      <td style="width:2%"></td>
      <td style="padding:12px;text-align:center;background:#FFA500;color:white;border-radius:8px;width:16%"><b style="font-size:20px">{sv.get('minor',0)}</b><br><small>Minor</small></td>
      <td style="width:2%"></td>
      <td style="padding:12px;text-align:center;background:#f8f9fa;border:2px solid {stc.get(rl,'#333')};color:{stc.get(rl,'#333')};border-radius:8px;width:24%"><b>{rl.upper()}</b><br><small>Risk</small><br><small>₹{tc:,}</small></td>
    </tr></table>
    {"<h4 style='color:#667eea'>📋 Detections</h4><table style='width:100%;border-collapse:collapse'><thead><tr style='background:#667eea;color:white'><th style='padding:7px'>#</th><th style='padding:7px;text-align:left'>Type</th><th style='padding:7px;text-align:left'>Model</th><th style='padding:7px'>Conf</th><th style='padding:7px'>Severity</th><th style='padding:7px;text-align:left'>Action</th><th style='padding:7px'>Cost</th></tr></thead><tbody>"+rows+"</tbody></table>" if dets else ""}
  </div>
  <div style="background:#f8f9fa;padding:10px;text-align:center;color:#999;font-size:11px">
    © {datetime.now().year} Srusti (1NT23AD052) — NMIT Bangalore
  </div>
</div></body></html>"""

def send_email(sender, password, recipient, dets, summary, risk,
               source="Unknown", pdf_bytes=None):
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🛣️ Road Damage Report — {summary.get('total',len(dets))} Issues [{datetime.now().strftime('%Y-%m-%d %H:%M')}]"
        msg["From"] = sender; msg["To"] = recipient
        msg.attach(MIMEText(_html(dets, summary, risk, source), "html"))
        final = msg
        if pdf_bytes:
            outer = MIMEMultipart("mixed")
            outer["Subject"]=msg["Subject"]; outer["From"]=sender; outer["To"]=recipient
            outer.attach(msg)
            part = MIMEBase("application","pdf"); part.set_payload(pdf_bytes)
            encoders.encode_base64(part)
            part.add_header("Content-Disposition",
                f'attachment; filename="RoadReport_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf"')
            outer.attach(part); final = outer
        with smtplib.SMTP_SSL("smtp.gmail.com",465,context=ssl.create_default_context()) as s:
            s.login(sender, password); s.sendmail(sender, recipient, final.as_string())
        return True, f"✅ Report sent to {recipient}"
    except smtplib.SMTPAuthenticationError:
        return False, "❌ Auth failed — use Gmail App Password"
    except Exception as e:
        return False, f"❌ {e}"
