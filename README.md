# DataForge AI 🔬

> **AI-powered Data Analytics Platform built with Python and Streamlit.**
> Upload any dataset — get instant quality reports, interactive dashboards, ML insights, and plain-English AI chat. No SQL. No code. No limits.

---

## 📸 Screenshots

| Upload & Profile | Data Quality Report | Dashboard |
|---|---|---|
| *(coming soon)* | *(coming soon)* | *(coming soon)* |

---

## ✨ Features

| Feature | Description |
|---|---|
| 📥 **Smart Upload** | CSV, Excel (multi-sheet), JSON — up to 200MB |
| 🧹 **Data Quality Report** | Per-column quality score, missing %, outlier detection, recommendations |
| 📊 **Auto Dashboard** | KPI cards, smart chart selection, correlation matrix, time-series detection |
| 🔬 **Deep Analysis** | Anomaly detection, distribution analysis, custom chart builder |
| 🤖 **AI Chat** | Ask questions in plain English — safe tool-calling, no code execution |
| 📄 **Reports** | Export cleaned data as CSV, PDF quality report, Excel with charts |

---

## 🛠️ Tech Stack

```
Python 3.11+       Core language
Streamlit          Web UI framework
Pandas / NumPy     Data processing
Plotly             Interactive charts
Scikit-learn       Anomaly detection, ML
Groq + Llama 3.3   AI chat engine
Tenacity           Retry logic
ReportLab          PDF export
Pydantic           Config management
```

---

## 🚀 Local Setup

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/dataforge-ai.git
cd dataforge-ai
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**

Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```

Get a free key at: console.groq.com

**5. Run the app**
```bash
streamlit run app.py
```

Open `http://localhost:8501`

---

## 📁 Project Structure

```
dataforge-ai/
│
├── app.py                        # Entry point only
├── requirements.txt
│
├── .streamlit/
│   ├── config.toml               # Theme + settings
│   └── secrets.toml              # API keys (gitignored)
│
├── pages/                        # One file = one page
│   ├── 1_📥_Data_Upload.py
│   ├── 2_🧹_Data_Quality.py
│   ├── 3_📊_Dashboard.py
│   ├── 4_🔬_Deep_Analysis.py
│   ├── 5_🤖_AI_Chat.py
│   └── 6_📄_Reports.py
│
├── core/                         # Business logic (no Streamlit)
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── data_profiler.py
│   ├── chart_engine.py
│   └── insight_engine.py
│
├── ai/                           # All LLM logic
│   ├── llm_client.py
│   ├── prompt_builder.py
│   ├── tool_dispatcher.py
│   └── response_parser.py
│
├── components/                   # Reusable UI widgets
│   ├── kpi_cards.py
│   ├── data_table.py
│   └── quality_report.py
│
├── config/
│   └── settings.py               # All config here
│
└── tests/
    ├── test_data_loader.py
    └── test_data_profiler.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Go to share.streamlit.io
3. Connect GitHub → select `dataforge-ai`
4. Add `GROQ_API_KEY` in Secrets
5. Click Deploy

---

## 🗺️ Roadmap

- [x] Phase 1 — Core platform (upload, quality, dashboard, AI chat)
- [ ] Phase 2 — ML features (anomaly detection, forecasting, clustering)
- [ ] Phase 3 — PDF/Excel export, multi-file join, saved dashboards
- [ ] Phase 4 — Database connectors (PostgreSQL, BigQuery, Snowflake)
- [ ] Phase 5 — FastAPI backend + React frontend

---

## 📝 License

© 2025 Shweta Mishra. All rights reserved.
This codebase is proprietary. Unauthorised copying, distribution, or use is strictly prohibited.

---

*Built by Shweta Mishra*
