import io
import streamlit as st
import plotly.io as pio
import pandas as pd
from ai.llm_client import LLMClient
from ai.prompt_builder import build_chat_system_prompt
from ai.response_parser import parse_tool_call
from ai.tool_dispatcher import dispatch
from core.data_profiler import profile_dataset
from components.kpi_cards import inject_global_css
import logging
logger = logging.getLogger(__name__)


# ── Global adaptive CSS (dark + light theme safe) ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important}
.block-container{padding-top:1.2rem!important}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1B2E,#0F2240)!important}
section[data-testid="stSidebar"] *{color:rgba(255,255,255,.85)!important}
section[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.12)!important}
/* adaptive card base */
.df-card{background:rgba(128,128,128,.06);border:1px solid rgba(128,128,128,.18);border-radius:12px;padding:16px 20px;margin-bottom:12px}
/* finding/risk/opp rows */
.risk-row{border-left:4px solid #ef4444;background:rgba(239,68,68,.07);padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:8px}
.opp-row{border-left:4px solid #10b981;background:rgba(16,185,129,.07);padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:8px}
.info-row{border-left:4px solid #3b82f6;background:rgba(59,130,246,.07);padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:8px}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="AI Chat — DataForge AI",
    page_icon="🤖",
    layout="wide"
)
inject_global_css()

if "df_active" not in st.session_state:
    st.warning("⚠️ No data loaded.")
    st.page_link("pages/1_📥_Data_Upload.py", label="← Go to Upload", icon="📥")
    st.stop()

df = st.session_state["df_active"]

from core.config import get_groq_key as _get_groq_key
groq_key = _get_groq_key()

if not groq_key:
    st.error("⚠️ GROQ_API_KEY not found in .streamlit/secrets.toml")
    st.stop()

client = LLMClient(api_key=groq_key)
system = build_chat_system_prompt(df)

st.title("🤖 AI Chat")
st.markdown(
    f"Ask anything about "
    f"**{st.session_state.get('filename', 'your dataset')}** "
    f"— {len(df):,} rows × {len(df.columns)} columns"
)
st.divider()

# ── Chat history ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ── Render history ─────────────────────────────────────────
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg.get("text"):
            st.markdown(msg["text"])
        if msg.get("fig_json"):
            try:
                st.plotly_chart(
                    pio.from_json(msg["fig_json"]),
                    use_container_width=True
                )
            except Exception:
                logger.debug("%s silent skip", exc_info=True)
        if msg.get("df_json"):
            try:
                st.dataframe(
                    pd.read_json(io.StringIO(msg["df_json"])),
                    use_container_width=True
                )
            except Exception:
                logger.debug("%s silent skip", exc_info=True)

# ── Input ──────────────────────────────────────────────────
prompt = st.chat_input(
    "e.g. 'Show top 10 sales by region' or 'Plot revenue over time'"
)

if prompt:
    st.session_state["messages"].append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            raw = client.chat_safe(
                messages=[{"role": "user", "content": prompt}],
                system=system
            )

        parsed = parse_tool_call(raw)

        if not parsed:
            txt = "Couldn't understand. Try: 'Show sales by region as bar chart'"
            st.markdown(txt)
            st.session_state["messages"].append({"role": "assistant", "text": txt})

        else:
            if parsed.get("explanation"):
                st.markdown(f"_{parsed['explanation']}_")

            result = dispatch(
                df,
                parsed["tool"],
                parsed["params"],
                parsed.get("explanation", "")
            )

            rec = {"role": "assistant", "text": parsed.get("explanation", "")}

            if not result.success:
                st.error(f"❌ {result.error}")
                rec["text"] = f"Error: {result.error}"

            else:
                if result.text_output:
                    st.markdown(result.text_output)
                    rec["text"] += "\n" + result.text_output

                if result.figure is not None:
                    st.plotly_chart(result.figure, use_container_width=True)
                    try:
                        rec["fig_json"] = pio.to_json(result.figure)
                    except Exception:
                        logger.debug("%s silent skip", exc_info=True)

                if result.dataframe is not None:
                    st.dataframe(
                        result.dataframe.head(50),
                        use_container_width=True
                    )
                    try:
                        rec["df_json"] = result.dataframe.head(50).to_json()
                    except Exception:
                        logger.debug("%s silent skip", exc_info=True)

            st.session_state["messages"].append(rec)

# ── Clear button ───────────────────────────────────────────
if st.session_state.get("messages"):
    if st.button("🗑️ Clear Chat"):
        st.session_state["messages"] = []
        st.rerun()
