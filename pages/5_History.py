"""History | Srusti 1NT23AD052 NMIT"""
import streamlit as st, pandas as pd, plotly.express as px
from utils.common import apply_css, copyright_footer
from utils.history_manager import load_history_df, clear_history, get_history_stats

st.set_page_config(page_title="History",page_icon="🕐",layout="wide")
apply_css()

st.markdown("""<div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:1.5rem;
border-radius:12px;color:white;text-align:center;margin-bottom:1rem">
<h2 style="margin:0">🕐 Detection History</h2>
<p style="margin:4px 0 0;opacity:.85">All detections across sessions · Filter · Export</p>
</div>""",unsafe_allow_html=True)

df = load_history_df()
if not df.empty:
    h = get_history_stats()
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total",h["total"])
    k2.metric("🔴 Severe",h["by_severity"].get("severe",0))
    k3.metric("🟠 Moderate",h["by_severity"].get("moderate",0))
    k4.metric("🟡 Minor",h["by_severity"].get("minor",0))
    st.divider()

    f1,f2,f3 = st.columns([2,2,1])
    with f1:
        types = ["All"] + (sorted(df["damage_type"].unique().tolist()) if "damage_type" in df.columns else [])
        ftype = st.selectbox("Filter by Type",types)
    with f2:
        fsev = st.selectbox("Filter by Severity",["All","minor","moderate","severe"])
    with f3:
        st.markdown("<br>",unsafe_allow_html=True)
        if st.button("🗑️ Clear All",use_container_width=True):
            clear_history(); st.rerun()

    filtered = df.copy()
    if ftype != "All" and "damage_type" in df.columns:
        filtered = filtered[filtered["damage_type"]==ftype]
    if fsev != "All" and "severity" in df.columns:
        filtered = filtered[filtered["severity"]==fsev]

    st.markdown(f"**{len(filtered)} records**")

    if not filtered.empty and "damage_type" in filtered.columns:
        bt = filtered["damage_type"].value_counts()
        fig = px.bar(x=bt.index.tolist(),y=bt.values.tolist(),
                     title="Filtered Detections by Type",color=bt.index.tolist(),
                     color_discrete_sequence=px.colors.qualitative.Set2,text_auto=True)
        fig.update_layout(showlegend=False,height=220,margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig,use_container_width=True)

    disp_cols = [c for c in ["timestamp","damage_type","confidence","severity","model","action","area"]
                 if c in filtered.columns]
    st.dataframe(
        filtered.sort_values("timestamp",ascending=False)[disp_cols]
        if "timestamp" in filtered.columns else filtered[disp_cols],
        use_container_width=True, hide_index=True)
    st.download_button("📥 Export CSV",filtered.to_csv(index=False),"history_export.csv","text/csv")
else:
    st.info("No history yet. Run detections to populate.")

copyright_footer()
