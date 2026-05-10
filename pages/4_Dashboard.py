"""Dashboard | Srusti 1NT23AD052 NMIT"""
import streamlit as st, plotly.express as px, plotly.graph_objects as go, pandas as pd
from utils.common import apply_css, copyright_footer
from utils.history_manager import get_history_stats, load_history_df

st.set_page_config(page_title="Dashboard",page_icon="📊",layout="wide")
apply_css()

st.markdown("""<div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:1.5rem;
border-radius:12px;color:white;text-align:center;margin-bottom:1rem">
<h2 style="margin:0">📊 Analytics Dashboard</h2>
<p style="margin:4px 0 0;opacity:.85">Real-time KPIs · Trends · Severity · Cost Estimates</p>
</div>""",unsafe_allow_html=True)

try: h=get_history_stats(); df=load_history_df()
except: h={"total":0,"by_type":{},"by_severity":{}}; df=pd.DataFrame()

k1,k2,k3,k4,k5=st.columns(5)
for col,val,label,grad in [
    (k1,h["total"],"Total","linear-gradient(135deg,#667eea,#764ba2)"),
    (k2,h["by_severity"].get("severe",0),"🔴 Severe","linear-gradient(135deg,#DC143C,#ff4444)"),
    (k3,h["by_severity"].get("moderate",0),"🟠 Moderate","linear-gradient(135deg,#FF6B6B,#ffa500)"),
    (k4,h["by_severity"].get("minor",0),"🟡 Minor","linear-gradient(135deg,#FFA500,#ffd700)"),
    (k5,h["by_type"].get("Pothole",0),"⚠️ Potholes","linear-gradient(135deg,#4facfe,#00f2fe)")]:
    col.markdown(f"""<div style="background:{grad};color:white;padding:1rem;border-radius:8px;text-align:center">
    <h2 style="margin:0">{val}</h2><small>{label}</small></div>""",unsafe_allow_html=True)

st.divider()

if h["total"]>0 and h["by_type"]:
    c1,c2=st.columns(2)
    with c1:
        fig=px.pie(values=list(h["by_type"].values()),names=list(h["by_type"].keys()),
                   title="Damage Type Distribution",hole=0.45,
                   color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=300,margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        sv=h["by_severity"]
        fig2=px.bar(x=["Minor","Moderate","Severe"],
                    y=[sv.get("minor",0),sv.get("moderate",0),sv.get("severe",0)],
                    title="Severity Breakdown",color=["Minor","Moderate","Severe"],
                    color_discrete_map={"Minor":"#FFA500","Moderate":"#FF6B6B","Severe":"#DC143C"},
                    text_auto=True)
        fig2.update_layout(showlegend=False,height=300,margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig2,use_container_width=True)

    if not df.empty and "timestamp" in df.columns and "damage_type" in df.columns:
        try:
            df["timestamp"]=pd.to_datetime(df["timestamp"]); df["date"]=df["timestamp"].dt.date
            daily=df.groupby(["date","damage_type"]).size().reset_index(name="count")
            fig3=px.line(daily,x="date",y="count",color="damage_type",
                         title="Daily Detection Trend",markers=True,
                         color_discrete_sequence=px.colors.qualitative.Set1)
            fig3.update_layout(height=280,margin=dict(t=50,b=10,l=10,r=10))
            st.plotly_chart(fig3,use_container_width=True)
        except: pass

    if not df.empty and "damage_type" in df.columns and "severity" in df.columns:
        try:
            pivot=df.groupby(["damage_type","severity"]).size().unstack(fill_value=0)
            for col in ["minor","moderate","severe"]:
                if col not in pivot.columns: pivot[col]=0
            pivot=pivot[["minor","moderate","severe"]]
            fig4=px.imshow(pivot.values,x=["Minor","Moderate","Severe"],y=pivot.index.tolist(),
                           color_continuous_scale="RdYlGn_r",
                           title="Severity × Damage Type Heatmap",text_auto=True)
            fig4.update_layout(height=260,margin=dict(t=50,b=10,l=10,r=10))
            st.plotly_chart(fig4,use_container_width=True)
        except: pass

    if not df.empty:
        st.markdown("### 🕐 Recent Detections")
        cols=[c for c in ["timestamp","damage_type","confidence","severity","model","action"] if c in df.columns]
        st.dataframe(df.sort_values("timestamp",ascending=False).head(25)[cols],
                     use_container_width=True,hide_index=True)
else:
    st.info("📂 No detection history yet. Run detections to populate the dashboard.")
    c1,c2=st.columns(2)
    with c1:
        fig=go.Figure(data=[go.Pie(
            labels=["Pothole","Long. Crack","Trans. Crack","Alligator Crack"],
            values=[35,25,20,20],hole=0.4,
            marker_colors=["#667eea","#764ba2","#f093fb","#f5576c"])])
        fig.update_layout(title="Sample: Damage Distribution",height=280,
                          margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig2=px.bar(x=["Minor","Moderate","Severe"],y=[40,35,25],title="Sample: Severity",
                    color=["Minor","Moderate","Severe"],
                    color_discrete_map={"Minor":"#FFA500","Moderate":"#FF6B6B","Severe":"#DC143C"},
                    text_auto=True)
        fig2.update_layout(showlegend=False,height=280,margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig2,use_container_width=True)
    st.caption("↑ Sample data — run detections to see real analytics.")

copyright_footer()
