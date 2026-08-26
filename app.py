import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Set page layout to wide
st.set_page_config(page_title="MPLADS Audit Engine", layout="wide")

st.title("MPLADS Infrastructure Audit Dashboard")
st.markdown("Advanced anomaly detection and risk scoring for government civic works.")

@st.cache_data
def load_data():
    return pd.read_csv('sih_audited_works_final.csv')

# Task 5: spinner only shows on first load; subsequent reruns return from cache instantly
with st.spinner("Loading audit data..."):
    df = load_data()

# --- SIDEBAR ---
# Task 5: one-line context header so the sidebar isn't cold-opened with bare dropdowns
st.sidebar.markdown("## MPLADS Audit Engine\nAI-powered anomaly detection across 77,576 sanctioned works.")
st.sidebar.divider()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Audits")

# 1. State Filter
state_list = ["All"] + sorted(df['State'].dropna().unique().tolist())
selected_state = st.sidebar.selectbox("Select State", state_list)

# Apply State Filter
if selected_state != "All":
    filtered_df = df[df['State'] == selected_state]
else:
    filtered_df = df.copy()

# 2. MP Filter
mp_list = ["All"] + sorted(filtered_df["Hon'ble Members of Parliament"].dropna().unique().tolist())
selected_mp = st.sidebar.selectbox("Select Member of Parliament", mp_list)

# Apply MP Filter
if selected_mp != "All":
    filtered_df = filtered_df[filtered_df["Hon'ble Members of Parliament"] == selected_mp]

st.sidebar.divider()
st.sidebar.info("Tip: Use these filters to isolate specific regions or politicians for targeted auditing.")

# --- KPI METRICS ---
st.subheader(f"System Overview: {selected_state if selected_state != 'All' else 'All India'}")
col1, col2, col3, col4 = st.columns(4)

high_risk_count = len(filtered_df[filtered_df['Total_Risk_Score'] >= 3])
stalled_count = filtered_df['is_stalled'].sum()
duplicate_count = filtered_df['is_duplicate_flag'].sum() if 'is_duplicate_flag' in filtered_df.columns else 0

with col1:
    st.metric(label="Total Projects", value=f"{len(filtered_df):,}")
with col2:
    st.metric(label="High Risk (Score 3+)", value=f"{high_risk_count:,}")
with col3:
    st.metric(label="Stalled (>1 Year)", value=f"{stalled_count:,}")
with col4:
    st.metric(label="Potential Duplicates", value=f"{duplicate_count:,}")

st.divider()

# --- INTERACTIVE CHARTS ---
st.subheader("Risk Distribution Analytics")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    high_risk_df = filtered_df[filtered_df['Total_Risk_Score'] >= 2]

    if not high_risk_df.empty:
        if selected_state == "All":
            risk_group = high_risk_df['State'].value_counts().reset_index()
            risk_group.columns = ['State', 'High Risk Projects']
            fig_1 = px.bar(risk_group.head(10), x='State', y='High Risk Projects', 
                           title='Top 10 States with Highest Risk Projects',
                           color='High Risk Projects', color_continuous_scale='Reds')
        else:
            risk_group = high_risk_df['work_category'].value_counts().reset_index()
            risk_group.columns = ['Work Category', 'High Risk Projects']
            fig_1 = px.bar(risk_group.head(10), x='Work Category', y='High Risk Projects', 
                           title=f'High Risk Projects by Category ({selected_state})',
                           color='High Risk Projects', color_continuous_scale='Reds')
        st.plotly_chart(fig_1, use_container_width=True)
    else:
        st.info("No high-risk projects found for this selection.")

with chart_col2:
    stalled_df = filtered_df[filtered_df['is_stalled'] == True]

    if not stalled_df.empty:
        stalled_status = stalled_df['Work Status'].value_counts().reset_index()
        stalled_status.columns = ['Work Status', 'Count']
        fig_status = px.bar(stalled_status, x='Work Status', y='Count',
                            title='Bottleneck Analysis (Where are stalled projects stuck?)',
                            color='Count', color_continuous_scale='Oranges')
        st.plotly_chart(fig_status, use_container_width=True)
    else:
        st.info("No stalled projects found for this selection.")

st.divider()

# --- EXPLAINABILITY ENGINE ---
def generate_audit_reasons(row):
    reasons = []
    
    if row.get('is_cost_outlier'):
        z_val = row.get('cost_zscore', 0)
        reasons.append(f"Outlier Cost ({z_val:+.1f}σ)")
        
    if row.get('is_speed_outlier'):
        days_sanc = row.get('days_to_sanction', 0)
        if days_sanc <= 12:
            reasons.append(f"Suspiciously Fast ({days_sanc}d approval)")
        else:
            reasons.append(f"Extreme Red Tape ({days_sanc}d approval)")
            
    if row.get('is_stalled'):
        days_old = row.get('days_since_sanction', 0)
        status_val = row.get('Work Status', 'Pending')
        reasons.append(f"Stalled {days_old}d ({status_val})")
        
    if row.get('is_duplicate_flag'):
        reasons.append("Near-Duplicate Contract")
        
    if row.get('is_ml_anomaly'):
        reasons.append("Multi-Variable Pattern Flag") # Removed "ML" verbiage
        
    return " | ".join(reasons) if reasons else "Normal Baseline"

# Task 1: cached wrapper — applies generate_audit_reasons row-by-row on the passed DataFrame.
# Running this on the full filtered_df (up to 77K rows) every rerun was the one genuine
# expensive repeatable computation. By moving it here, keyed on df content hash, it:
#   (a) only runs on the priority subset (score >= 3), which is much smaller, AND
#   (b) returns from cache instantly if the same filter is revisited.
# NOTE: @st.cache_data hashes the DataFrame input, so this is correctly invalidated
# whenever the filtered selection changes.
@st.cache_data
def compute_audit_reasons(df):
    return df.apply(generate_audit_reasons, axis=1)

# --- HIGH RISK AUDIT TABLE ---
st.subheader("Priority Audit Queue (Risk Score 3 & 4)")
st.caption("Detailed breakdown explaining why each project was flagged by the anomaly engine.")

# Build priority subset first, then compute reasons only on that smaller DataFrame
# (original code computed reasons on all of filtered_df, then discarded non-priority rows).
priority_df = filtered_df[filtered_df['Total_Risk_Score'] >= 3].sort_values(by='Total_Risk_Score', ascending=False)
priority_df = priority_df.copy()  # explicit copy to avoid SettingWithCopyWarning
priority_df['Flagged Reasons'] = compute_audit_reasons(priority_df)

if not priority_df.empty:
    display_cols = [
        'Total_Risk_Score',
        'Flagged Reasons',
        'State', 
        "Hon'ble Members of Parliament", 
        'work_category', 
        'Work description', 
        'Sanction_Amount'
    ]
    
    st.dataframe(
        priority_df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Total_Risk_Score": st.column_config.NumberColumn("Risk Score", format="%d"),
            "Sanction_Amount": st.column_config.NumberColumn("Sanctioned Amount", format="₹%d"),
            "Flagged Reasons": st.column_config.TextColumn("Audit Flags & Justification", width="large"),
            "Work description": st.column_config.TextColumn("Description", width="large")
        }
    )

    # Task 5: download button — one line, high demo impact
    st.download_button(
        label="⬇️ Download Flagged Projects (CSV)",
        data=priority_df[display_cols].to_csv(index=False).encode('utf-8'),
        file_name="mplads_priority_audit_queue.csv",
        mime="text/csv"
    )
else:
    st.success("No high-risk projects found for this specific filter.")

# --- FORENSIC AUDIT INVESTIGATOR ---
st.divider()
st.subheader("Deep Dive: Forensic Audit Investigator")
st.markdown("Select a high-risk project from the table above to generate a forensic audit explanation.")

# Task 3: Isolated, cached API caller — keyed on the individual scalar fields that
# build the prompt, NOT on the DataFrame row (unhashable) or Select_Label (collision-prone).
# @st.cache_data only caches successful returns; exceptions propagate to the caller and
# are NOT cached by Streamlit, satisfying the "don't cache failures" requirement.
#
# KNOWN TRADE-OFF: If sih_audited_works_final.csv is regenerated between demo runs with
# different risk data for the same project, this cache will serve stale responses until
# Streamlit's cache is cleared (streamlit cache clear, or app restart). Acceptable for
# single-demo use.
@st.cache_data
def _call_gemini(state, mp, category, description, sanction_amount, work_status, risk_score, flagged_reasons):
    from google import genai
    client = genai.Client()
    prompt = f"""
    Write a formal, objective government audit report analyzing the following infrastructure project. 
    Do not use words like "AI", "Machine Learning", or conversational language. Output a structured, professional risk assessment.
    
    Project Details:
    - State: {state}
    - MP: {mp}
    - Category: {category}
    - Description: {description}
    - Sanctioned Amount: INR {sanction_amount}
    - Current Status: {work_status}
    
    Anomaly Flags Triggered:
    - Risk Score: {risk_score}/5
    - Specific Reasons: {flagged_reasons}
    
    Provide a 2-3 paragraph risk analysis focusing on potential administrative, financial, or procurement irregularities (e.g., rubber-stamping, tender splitting, abandoned works) based strictly on these flags.
    """
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )
    return response.text

if not priority_df.empty:
    priority_df['Select_Label'] = priority_df['State'] + " | " + priority_df["Hon'ble Members of Parliament"] + " | ₹" + priority_df['Sanction_Amount'].astype(int).astype(str)
    
    selected_project_label = st.selectbox("Select Project to Investigate:", priority_df['Select_Label'])
    
    if st.button("Generate Forensic Audit Report"):
        project_data = priority_df[priority_df['Select_Label'] == selected_project_label].iloc[0]
        
        with st.spinner("Compiling report data..."):
            # Task 2: Pre-flight check for missing key — most reliable signal for this
            # specific case. Avoids depending on SDK exception messages, which vary by
            # version and may change. If the key IS set but invalid, the API call below
            # returns a ClientError(401), handled separately.
            if not os.environ.get('GEMINI_API_KEY'):
                st.error("⚠️ Gemini API key not configured. Set the GEMINI_API_KEY environment variable and restart the app.")
                print("ERROR: GEMINI_API_KEY environment variable is not set.")
            else:
                report_text = None
                try:
                    report_text = _call_gemini(
                        state=str(project_data['State']),
                        mp=str(project_data["Hon'ble Members of Parliament"]),
                        category=str(project_data['work_category']),
                        description=str(project_data['Work description']),
                        sanction_amount=float(project_data['Sanction_Amount']),
                        work_status=str(project_data['Work Status']),
                        risk_score=int(project_data['Total_Risk_Score']),
                        flagged_reasons=str(project_data['Flagged Reasons'])
                    )
                except Exception as e:
                    # Log full exception for post-demo debugging — never shown to user.
                    print(f"ERROR [{type(e).__name__}]: {e}")

                    # Task 2: Classify using SDK exception types (confirmed from SDK source:
                    # google/genai/errors.py). ClientError covers all 4xx; ServerError covers
                    # 5xx. Network-level errors (httpx.ConnectError, httpx.TimeoutException)
                    # are NOT subclasses of genai errors — they fall to the generic branch.
                    # Collapsing network errors into "unexpected" is noted as a known gap.
                    try:
                        from google.genai.errors import ClientError, ServerError
                        if isinstance(e, ClientError):
                            if getattr(e, 'code', None) == 429:
                                st.error("⚠️ Report generation is rate-limited right now. Please wait a few seconds and try again.")
                            else:
                                # 401, 403, or other 4xx — treat as auth/key issue
                                st.error("⚠️ Gemini API key not configured or invalid. Check your GEMINI_API_KEY environment variable and restart the app.")
                        elif isinstance(e, ServerError):
                            st.error("⚠️ Could not reach the Gemini API (server error). Please try again in a moment.")
                        else:
                            # Covers network errors, unexpected exceptions, etc.
                            st.error("⚠️ Report generation failed unexpectedly. Please try again or proceed without the AI report for this project.")
                    except ImportError:
                        # google-genai not installed or errors module unavailable
                        st.error("⚠️ Report generation failed unexpectedly. Please try again or proceed without the AI report for this project.")

                if report_text is not None:
                    st.success("Analysis Complete")
                    with st.expander("Official Audit Investigation Report", expanded=True):
                        st.markdown(report_text)