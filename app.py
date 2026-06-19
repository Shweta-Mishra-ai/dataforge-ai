"""
app.py — DataForge AI
Professional landing page with client config, logo upload, and file upload.
Client name + logo collected HERE — not buried in page 8.
"""
import streamlit as st
import os

st.set_page_config(
    page_title="DataForge AI — Professional Data Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.session_manager import init_session, set_dataframe
from core.data_loader import load_file
from core.data_profiler import profile_dataset

init_session()

# ── Premium CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
.block-container { padding-top: 0 !important; padding-bottom: 1rem !important; }

/* Hero */
.hero-wrap {
    background: linear-gradient(135deg, #0D1B2E 0%, #1B3A6B 60%, #0D2137 100%);
    padding: 52px 48px 44px;
    border-radius: 0 0 24px 24px;
    margin: -1rem -1rem 0 -1rem;
    position: relative; overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute; top: -80px; right: -60px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(37,99,168,0.25) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}
.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);
    color: rgba(255,255,255,0.85); font-size: 11px; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 5px 14px; border-radius: 20px; margin-bottom: 20px;
}
.hero-title {
    font-size: clamp(28px, 4vw, 44px); font-weight: 900; color: #FFFFFF;
    line-height: 1.15; margin: 0 0 14px;
}
.hero-title span { color: #60A5FA; }
.hero-sub {
    font-size: 16px; color: rgba(255,255,255,0.65);
    font-weight: 400; max-width: 560px; line-height: 1.6; margin: 0 0 32px;
}
.hero-stats {
    display: flex; gap: 32px; flex-wrap: wrap;
}
.hero-stat-val { font-size: 22px; font-weight: 800; color: #60A5FA; }
.hero-stat-lbl { font-size: 11px; color: rgba(255,255,255,0.5); text-transform: uppercase; letter-spacing: 0.06em; }

/* Feature cards */
.feat-card {
    background: white; border: 1px solid #E2E8F0;
    border-radius: 16px; padding: 24px 22px;
    transition: box-shadow 0.2s, transform 0.2s;
    height: 100%;
}
.feat-card:hover { box-shadow: 0 8px 32px rgba(0,0,0,0.08); transform: translateY(-2px); }
.feat-icon { font-size: 28px; margin-bottom: 12px; }
.feat-title { font-size: 15px; font-weight: 700; color: #0F172A; margin-bottom: 8px; }
.feat-desc { font-size: 13px; color: #64748B; line-height: 1.6; }

/* Domain badges */
.domain-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 8px 16px; border-radius: 20px;
    font-size: 12px; font-weight: 700; letter-spacing: 0.05em;
    margin: 4px;
}

/* Upload zone */
.upload-card {
    background: linear-gradient(135deg, #F0F7FF 0%, #EFF6FF 100%);
    border: 2px dashed #93C5FD; border-radius: 16px;
    padding: 32px 28px; text-align: center;
}
.upload-title { font-size: 18px; font-weight: 700; color: #1E3A5F; margin-bottom: 8px; }
.upload-sub   { font-size: 13px; color: #64748B; margin-bottom: 20px; }

/* Config card */
.config-card {
    background: white; border: 1px solid #E2E8F0;
    border-radius: 16px; padding: 24px; margin-bottom: 16px;
}
.config-title {
    font-size: 13px; font-weight: 700; color: #0F172A;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin-bottom: 16px; padding-bottom: 10px;
    border-bottom: 2px solid #E2E8F0;
}

/* Step strip */
.step-strip {
    display: flex; gap: 0; align-items: stretch;
    background: #0F172A; border-radius: 12px;
    overflow: hidden; margin: 24px 0;
}
.step-item {
    flex: 1; padding: 20px 18px; text-align: center;
    border-right: 1px solid rgba(255,255,255,0.08);
    position: relative;
}
.step-item:last-child { border-right: none; }
.step-num {
    width: 32px; height: 32px;
    background: #2563A8; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 800; color: white;
    margin: 0 auto 10px;
}
.step-lbl { font-size: 13px; font-weight: 700; color: white; margin-bottom: 4px; }
.step-sub { font-size: 11px; color: rgba(255,255,255,0.45); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2E 0%, #0F2240 100%) !important;
}
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stButton button {
    color: white !important;
}
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }

/* Ready banner */
.ready-banner {
    background: linear-gradient(135deg, #064E3B, #065F46);
    border-radius: 14px; padding: 20px 24px; margin-top: 16px;
    border: 1px solid #059669;
}
.ready-title { font-size: 18px; font-weight: 800; color: #34D399; margin-bottom: 8px; }
.ready-sub   { font-size: 13px; color: rgba(255,255,255,0.7); }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 20px 0 12px; text-align: center;">
        <div style="font-size: 22px; font-weight: 900; color: white; letter-spacing: -0.5px;">
            🔬 DataForge AI
        </div>
        <div style="font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase;
                    letter-spacing: 0.1em; margin-top: 4px;">
            Advanced Analytics Platform
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    df_loaded = st.session_state.get("df") is not None
    if df_loaded:
        fname = st.session_state.get("filename", "Dataset")
        nrows = len(st.session_state["df"])
        ncols = len(st.session_state["df"].columns)
        st.markdown(f"""
        <div style="background:rgba(52,211,153,0.12); border:1px solid rgba(52,211,153,0.3);
                    border-radius:10px; padding:12px 14px; margin-bottom:12px;">
            <div style="font-size:11px; font-weight:700; color:#34D399;
                        letter-spacing:.07em; text-transform:uppercase; margin-bottom:5px;">
                ✅ Data Loaded
            </div>
            <div style="font-size:13px; font-weight:600; color:white; margin-bottom:2px;">
                {fname[:28]}
            </div>
            <div style="font-size:11px; color:rgba(255,255,255,0.45);">
                {nrows:,} rows · {ncols} columns
            </div>
        </div>
        """, unsafe_allow_html=True)

    nav_items = [
        ("🏠", "Home",           True),
        ("📊", "Dashboard",      df_loaded),
        ("🔍", "Data Quality",   df_loaded),
        ("💡", "Business Insights", df_loaded),
        ("📈", "Deep EDA",       df_loaded),
        ("🤖", "ML Predictions", df_loaded),
        ("📄", "Reports",        df_loaded),
        ("💬", "AI Chat",        df_loaded),
    ]
    for icon, label, active in nav_items:
        color = "rgba(255,255,255,0.85)" if active else "rgba(255,255,255,0.25)"
        st.markdown(f"""
        <div style="padding:7px 12px; border-radius:8px; margin-bottom:2px;
                    color:{color}; font-size:13px; font-weight:500;">
            {icon} &nbsp; {label}
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    try:
        _groq_key = st.secrets.get("GROQ_API_KEY", "") or ""
    except Exception:
        _groq_key = os.environ.get("GROQ_API_KEY", "") or ""

    if not _groq_key.strip():
        st.markdown("""
        <div style="background:rgba(255,209,102,0.08); border:1px solid rgba(255,209,102,0.2);
                    border-radius:8px; padding:8px 12px; margin-bottom:8px;">
            <div style="font-size:10px; font-weight:700; color:#ffd166;
                        letter-spacing:.07em; text-transform:uppercase;">⚡ AI Narratives OFF</div>
            <div style="font-size:10px; color:rgba(255,255,255,0.35); margin-top:2px;">
                Add GROQ_API_KEY in secrets.toml<br>to enable AI-generated insights
            </div>
        </div>
        """, unsafe_allow_html=True)

    if df_loaded:
        st.markdown("<div style='font-size:11px; color:rgba(255,255,255,0.35);'>Use the pages in the sidebar above to navigate.</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <div class="hero-tag">◆ Professional Data Analysis Platform</div>
    <div class="hero-title">Turn Raw Data Into<br><span>Board-Ready Reports</span></div>
    <div class="hero-sub">
        Upload any dataset. Get deep statistical analysis, business insights,
        and a professionally designed PDF report — ready to send to clients.
    </div>
    <div class="hero-stats">
        <div>
            <div class="hero-stat-val">14+</div>
            <div class="hero-stat-lbl">Analysis Types</div>
        </div>
        <div>
            <div class="hero-stat-val">4</div>
            <div class="hero-stat-lbl">Business Domains</div>
        </div>
        <div>
            <div class="hero-stat-val">PDF</div>
            <div class="hero-stat-lbl">Client-Grade Reports</div>
        </div>
        <div>
            <div class="hero-stat-val">200MB</div>
            <div class="hero-stat-lbl">Max File Size</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  HOW IT WORKS
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="step-strip">
    <div class="step-item">
        <div class="step-num">1</div>
        <div class="step-lbl">Configure</div>
        <div class="step-sub">Enter client name & logo</div>
    </div>
    <div class="step-item">
        <div class="step-num">2</div>
        <div class="step-lbl">Upload</div>
        <div class="step-sub">CSV, Excel, or JSON</div>
    </div>
    <div class="step-item">
        <div class="step-num">3</div>
        <div class="step-lbl">Analyse</div>
        <div class="step-sub">Auto deep analysis</div>
    </div>
    <div class="step-item">
        <div class="step-num">4</div>
        <div class="step-lbl">Download</div>
        <div class="step-sub">Board-ready PDF</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT — CONFIG LEFT, UPLOAD RIGHT
# ═══════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.markdown("### ⚙️ Report Configuration")
    st.caption("Set up once — used in all reports automatically")

    # ── Client name ───────────────────────────────────────────────────
    client_name = st.text_input(
        "Client / Company Name",
        value=st.session_state.get("client_name", ""),
        placeholder="e.g. Acme Corporation",
        help="Appears on the PDF cover page and all report headers",
        key="input_client_name"
    )
    if client_name:
        st.session_state["client_name"] = client_name

    # ── Report title ───────────────────────────────────────────────────
    report_title = st.text_input(
        "Report Title",
        value=st.session_state.get("report_title", ""),
        placeholder="e.g. Q1 2026 HR Analytics Report",
        key="input_report_title"
    )
    if report_title:
        st.session_state["report_title"] = report_title

    # ── Logo upload ────────────────────────────────────────────────────
    logo_file = st.file_uploader(
        "Company Logo (optional)",
        type=["png", "jpg", "jpeg", "svg"],
        help="PNG with transparent background recommended. Appears on PDF cover.",
        key="logo_uploader"
    )
    if logo_file:
        logo_bytes = logo_file.read()
        logo_ext   = logo_file.name.split(".")[-1].lower()
        st.session_state["logo_bytes"] = logo_bytes
        st.session_state["logo_ext"]   = logo_ext
        st.session_state["logo_path"]  = ""
        st.success(f"✅ Logo saved: {logo_file.name}")

    # ── Analyst name ───────────────────────────────────────────────────
    analyst = st.text_input(
        "Prepared By (Analyst Name)",
        value=st.session_state.get("analyst_name", ""),
        placeholder="Your name or firm",
        key="input_analyst"
    )
    if analyst:
        st.session_state["analyst_name"] = analyst

    # Config summary
    if client_name or report_title:
        st.markdown("""
        <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:10px;
                    padding:14px 16px; margin-top:8px;">
            <div style="font-size:11px; font-weight:700; color:#166534;
                        letter-spacing:.07em; text-transform:uppercase; margin-bottom:6px;">
                ✅ Report Config Saved
            </div>
            <div style="font-size:12px; color:#374151;">
                This will appear on the PDF cover page automatically.
                No need to re-enter in the Reports section.
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown("### 📂 Upload Your Data")
    st.caption("Supports CSV · Excel (.xlsx/.xls, multi-sheet) · JSON — up to 200 MB")

    uploaded = st.file_uploader(
        label="Drop file here or click to browse",
        type=["csv", "xlsx", "xls", "json"],
        label_visibility="collapsed",
        key="main_uploader"
    )

    if not uploaded:
        st.markdown("""
        <div class="upload-card">
            <div style="font-size:40px; margin-bottom:12px;">📁</div>
            <div class="upload-title">Drop your file here</div>
            <div class="upload-sub">
                CSV · Excel (multi-sheet supported) · JSON<br>
                All analysis runs in your session — data is never stored on any server.
            </div>
            <div style="display:flex; gap:8px; justify-content:center; flex-wrap:wrap; margin-top:16px;">
                <span style="background:#DBEAFE; color:#1D4ED8; font-size:11px; font-weight:700;
                             padding:4px 12px; border-radius:12px;">HR Analytics</span>
                <span style="background:#FEF3C7; color:#B45309; font-size:11px; font-weight:700;
                             padding:4px 12px; border-radius:12px;">E-Commerce</span>
                <span style="background:#D1FAE5; color:#065F46; font-size:11px; font-weight:700;
                             padding:4px 12px; border-radius:12px;">Sales</span>
                <span style="background:#EDE9FE; color:#5B21B6; font-size:11px; font-weight:700;
                             padding:4px 12px; border-radius:12px;">Finance</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if uploaded:
        with st.spinner(f"Loading {uploaded.name}..."):
            result = load_file(uploaded)

        if not result.success:
            st.error(result.error)
            st.stop()

        # Sheet selector
        if result.sheet_names and len(result.sheet_names) > 1:
            st.success(f"Excel — {len(result.sheet_names)} sheets found")
            sel_sheet = st.selectbox("Select sheet:", result.sheet_names)
            with st.spinner(f"Loading '{sel_sheet}'..."):
                result = load_file(uploaded, sheet_name=sel_sheet)
            if not result.success:
                st.error(result.error)
                st.stop()

        df = result.df
        set_dataframe(df, result.filename, result.file_size_mb)
        st.session_state["profile"] = None

        # Profile
        with st.spinner("Analysing data quality..."):
            try:
                sample = df.sample(n=min(10000, len(df)), random_state=42) if len(df) > 10000 else df
                profile = profile_dataset(sample)
                st.session_state["profile"] = profile
                qual = getattr(profile, "overall_quality_score", "—")
                grade = getattr(profile, "data_quality_grade", "")
                miss = getattr(profile, "missing_pct", df.isna().mean().mean() * 100)
            except Exception:
                profile = None
                qual = "—"
                grade = ""
                miss = df.isna().mean().mean() * 100

        # Auto-set report title if not set
        if not st.session_state.get("report_title"):
            clean = result.filename
            for ext in [".csv", ".xlsx", ".xls", ".json"]:
                clean = clean.replace(ext, "")
            st.session_state["report_title"] = f"Data Analysis Report — {clean.replace('_',' ').strip()}"

        # ── Ready banner ───────────────────────────────────────────────
        st.markdown(f"""
        <div class="ready-banner">
            <div class="ready-title">✅ {result.filename} Loaded</div>
            <div class="ready-sub">
                {len(df):,} rows · {len(df.columns)} columns ·
                Quality: {qual}/100 {f'(Grade {grade})' if grade else ''} ·
                Missing: {miss:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── CTA buttons ───────────────────────────────────────────────
        b1, b2, b3 = st.columns(3)
        with b1:
            st.page_link("pages/3_#L01f4ca_Dashboard.py", label="📊 Explore Dashboard", use_container_width=True)
        with b2:
            st.page_link("pages/2_#L01f9f9_Data_Quality.py", label="🔍 Quality Report", use_container_width=True)
        with b3:
            st.page_link("pages/8_Reports.py", label="📄 Generate PDF", use_container_width=True)

        # Warnings
        if result.warnings:
            with st.expander(f"⚠️ {len(result.warnings)} data warnings"):
                for w in result.warnings:
                    st.warning(w)

# ═══════════════════════════════════════════════════════════════════════
#  FEATURE STRIP
# ═══════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### What DataForge AI Does")

f1, f2, f3, f4 = st.columns(4, gap="medium")
features = [
    ("📊", "Instant Quality Audit",
     "Detect missing values, duplicates, outliers, and skewed distributions. Get a scored quality report with remediation steps."),
    ("🧠", "Deep Statistical Analysis",
     "Distributions, normality tests, correlations, cohort comparisons, and logistic regression drivers — all automated."),
    ("💡", "Business Intelligence",
     "Auto-detects HR, Sales, E-commerce, or Finance domain. Generates domain-specific insights with Problem → Action → Impact structure."),
    ("📄", "Board-Ready PDF Reports",
     "Professional multi-page reports with cover page, executive summary, charts, and recommendations — client-branded with your logo."),
]
for col, (icon, title, desc) in zip([f1, f2, f3, f4], features):
    with col:
        st.markdown(f"""
        <div class="feat-card">
            <div class="feat-icon">{icon}</div>
            <div class="feat-title">{title}</div>
            <div class="feat-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  DOMAIN BADGES
# ═══════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding: 24px 0;">
    <div style="font-size:13px; font-weight:700; color:#6B7280; text-transform:uppercase;
                letter-spacing:0.1em; margin-bottom:16px;">
        Supported Business Domains
    </div>
    <span class="domain-badge" style="background:#DBEAFE; color:#1D4ED8;">
        👥 HR & People Analytics
    </span>
    <span class="domain-badge" style="background:#FEF3C7; color:#B45309;">
        🛒 E-Commerce Analytics
    </span>
    <span class="domain-badge" style="background:#D1FAE5; color:#065F46;">
        📈 Sales Performance
    </span>
    <span class="domain-badge" style="background:#F3E8FF; color:#6B21A8;">
        💰 Finance & Accounting
    </span>
    <span class="domain-badge" style="background:#F1F5F9; color:#475569;">
        📋 General Business
    </span>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center; padding:20px 0 8px; border-top:1px solid #E2E8F0; margin-top:16px;">
    <span style="font-size:12px; color:#9CA3AF;">
        DataForge AI · Advanced Analytics Platform · All data processed locally in your session
    </span>
</div>
""", unsafe_allow_html=True)
