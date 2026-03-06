import streamlit as st


def inject_global_css():
    st.markdown("""
    <style>
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(155px, 1fr));
        gap: 14px;
        margin: 16px 0 28px;
    }
    .kpi-card {
        background: #0e0f1a;
        border: 1px solid #1e2035;
        border-radius: 12px;
        padding: 18px 16px 14px;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: var(--kpi-accent, #4f8ef7);
    }
    .kpi-icon {
        font-size: 18px;
        margin-bottom: 6px;
        display: block;
    }
    .kpi-label {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #5a5f82;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 800;
        color: #dde1f5;
        line-height: 1.1;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-sub {
        font-size: 11px;
        color: #5a5f82;
        margin-top: 5px;
    }
    .q-bar-bg {
        background: #1c1d32;
        border-radius: 4px;
        height: 6px;
        margin-top: 8px;
        overflow: hidden;
    }
    .q-bar-fill {
        height: 100%;
        border-radius: 4px;
    }
    .insight-card {
        background: #0e0f1a;
        border: 1px solid #1e2035;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-left: 3px solid var(--ic, #4f8ef7);
    }
    .insight-title {
        font-size: 13px;
        font-weight: 700;
        color: #dde1f5;
        margin-bottom: 4px;
    }
    .insight-body {
        font-size: 12px;
        color: #6b7299;
        line-height: 1.7;
    }
    </style>
    """, unsafe_allow_html=True)


def kpi_grid(cards: list):
    html = '<div class="kpi-grid">'
    for c in cards:
        accent = c.get("accent", "#4f8ef7")
        icon   = c.get("icon", "")
        sub    = c.get("sub", "")
        html += f"""
        <div class="kpi-card" style="--kpi-accent:{accent}">
            {"<span class='kpi-icon'>" + icon + "</span>" if icon else ""}
            <div class="kpi-label">{c['label']}</div>
            <div class="kpi-value">{c['value']}</div>
            {"<div class='kpi-sub'>" + sub + "</div>" if sub else ""}
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def quality_score_banner(score: float):
    if score >= 80:
        colour = "#22d3a5"
        label  = "Good"
    elif score >= 60:
        colour = "#f7934f"
        label  = "Fair"
    else:
        colour = "#f77070"
        label  = "Poor"

    st.markdown(f"""
    <div class="kpi-card" style="--kpi-accent:{colour}; max-width:260px; margin-bottom:20px">
        <div class="kpi-label">Overall Data Quality</div>
        <div class="kpi-value" style="color:{colour}">
            {score}<span style="font-size:16px">/100</span>
        </div>
        <div class="kpi-sub">{label} quality</div>
        <div class="q-bar-bg">
            <div class="q-bar-fill" style="width:{score}%; background:{colour}"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def insight_card(title: str, body: str, type_: str = "info"):
    colors = {
        "info":     "#4f8ef7",
        "warning":  "#f7934f",
        "positive": "#22d3a5",
        "negative": "#f77070",
    }
    colour = colors.get(type_, "#4f8ef7")
    st.markdown(f"""
    <div class="insight-card" style="--ic:{colour}">
        <div class="insight-title">{title}</div>
        <div class="insight-body">{body}</div>
    </div>
    """, unsafe_allow_html=True)
