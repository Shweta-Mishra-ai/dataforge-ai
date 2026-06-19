<div align="center">

<img src="https://img.shields.io/badge/DataForge_AI-v2.0-1B4FD8?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0yMSAxNlY4YTIgMiAwIDAgMC0xLTEuNzNsLTctNGEyIDIgMCAwIDAtMiAwbC03IDRBMiAyIDAgMCAwIDMgOHY4YTIgMiAwIDAgMCAxIDEuNzNsNyA0YTIgMiAwIDAgMCAyIDBsNy00QTIgMiAwIDAgMCAyMSAxNnoiLz48L3N2Zz4=" alt="DataForge AI">

# DataForge AI

**Enterprise-grade AI analytics platform — upload any dataset, get instant intelligence.**

[![CI](https://github.com/Shweta-Mishra-ai/dataforge-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Shweta-Mishra-ai/dataforge-ai/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Ruff](https://img.shields.io/badge/lint-ruff-FCC21B?style=flat)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-23%20passing-059669?style=flat&logo=pytest)](https://github.com/Shweta-Mishra-ai/dataforge-ai/tree/main/tests)
[![License](https://img.shields.io/badge/license-Proprietary-6B7280?style=flat)](LICENSE)

[**Live Demo**](https://dataforge-ai.streamlit.app) · [**Report a Bug**](https://github.com/Shweta-Mishra-ai/dataforge-ai/issues) · [**Request a Feature**](https://github.com/Shweta-Mishra-ai/dataforge-ai/issues)

</div>

---

## What is DataForge AI?

DataForge AI transforms raw CSV, Excel, or JSON files into executive-ready intelligence in seconds. No SQL. No code. No BI tool subscription.

Upload a dataset → get a **17-page branded PDF report** with domain-aware insights, statistical analysis, ML predictions, and AI-written narratives — ready to deliver to a client.

```
Upload CSV/Excel/JSON
        ↓
Auto domain detection (HR · Sales · Finance · E-commerce)
        ↓
Data quality scoring · Outlier detection · Deduplication
        ↓
Statistical analysis · Correlation matrix · ML predictions
        ↓
AI narratives (Groq Llama 3.3 70B) · Industry benchmarks
        ↓
Branded PDF report + Health report — client-ready
```

---

## Features

### 📊 Smart Analytics Engine
| Capability | Details |
|---|---|
| **Domain Detection** | Auto-identifies HR, Sales, Finance, E-commerce datasets from column names |
| **Data Quality Scoring** | 0–100 score: 60% completeness · 30% deduplication · 10% column health |
| **Statistical Analysis** | Normality tests · Pearson/Spearman correlations with r² · Outlier detection |
| **Business Intelligence** | Cohort analysis · Kruskal-Wallis significance testing · Root cause identification |
| **ML Predictions** | Auto model selection · Cross-validation · SHAP feature importance · Overfit detection |

### 📄 Report Generation
| Report Type | Pages | What's Inside |
|---|---|---|
| **Analysis Report** | 17 pages | Executive summary · Industry benchmarks · Attrition deep dive · 5 smart charts · Recommendations |
| **Health Report** | 5 pages | Quality score · Business insights · Descriptive stats · Column quality table · Correlation matrix |

### 🧠 AI-Powered Insights
- **6 deep HR insights** — tenure cohort risk, overwork detection, salary band attrition, promotion gap, flight risk segment
- **Groq Llama 3.3 70B** for chart narratives and executive summaries
- **Rule-based fallback** — full insight generation even without API key

### 🎨 Premium PDF Output
- **Carlito font** (Calibri-compatible) — professional typography
- **Domain-aware cover** — HR badge, Finance badge, E-commerce badge
- **Smart charts** — mean for score metrics, sum for revenue metrics, with reference lines
- **5 color themes** — Corporate Light, Dark Tech, Executive Green, Ocean Blue, Slate Gray

---

## Tech Stack

```
Layer           Technology              Purpose
─────────────── ─────────────────────── ──────────────────────────────────────
UI              Streamlit 1.32+         Multi-page web app
Data            Pandas 2.0 · NumPy      Processing, profiling, cleaning
Visualization   Plotly · Matplotlib     Interactive charts + PDF charts
PDF             ReportLab               Branded report generation
ML              Scikit-learn · XGBoost  Auto model selection, SHAP
Stats           SciPy · Statsmodels     Normality, correlation, significance
AI              Groq + Llama 3.3 70B    Narrative generation
Fonts           Carlito · Liberation    Premium PDF typography
Quality         Ruff · Pytest           Lint + 23 unit tests
CI              GitHub Actions          Automated on every push
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Git

### 1. Clone
```bash
git clone https://github.com/Shweta-Mishra-ai/dataforge-ai.git
cd dataforge-ai
```

### 2. Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Secrets
Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```

> **Free key:** [console.groq.com](https://console.groq.com) — 30 req/min on the free tier.
> App works without a key — AI narratives fall back to computed rule-based output.

### 5. Run
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## Project Structure

```
dataforge-ai/
│
├── app.py                          # Home page — file upload, client config
├── conftest.py                     # Pytest path setup
├── requirements.txt
│
├── .github/
│   └── workflows/
│       └── ci.yml                  # Ruff + Pytest on every push
│
├── .streamlit/
│   ├── config.toml                 # Theme + performance settings
│   └── secrets.toml                # API keys (gitignored)
│
├── pages/                          # One file = one Streamlit page
│   ├── 2_Data_Quality.py           # Quality score, column analysis
│   ├── 3_Dashboard.py              # KPI cards, smart charts
│   ├── 4_Business_Insights.py      # BI engine, cohort analysis
│   ├── 5_ML_Predictions.py         # Auto ML, SHAP, forecasting
│   ├── 6_Deep_EDA.py               # Statistical deep dive
│   ├── 7_Business_Intel.py         # Domain intelligence
│   ├── 8_Reports.py                # PDF report generation ← main output
│   ├── 9_AI_Chat.py                # Natural language Q&A
│   ├── 10_Deep_Analysis.py         # Advanced analytics
│   └── 11_Health_Report.py         # Data health + business insights PDF
│
├── core/                           # Business logic — zero Streamlit imports
│   ├── data_loader.py              # Multi-format loader, smart dtype inference
│   ├── data_profiler.py            # Quality scoring, outlier detection
│   ├── data_validator.py           # Schema validation
│   ├── data_cleaner.py             # Deduplication, imputation
│   ├── chart_engine.py             # Plotly chart builder, smart aggregation
│   ├── chart_exporter.py           # Matplotlib → PNG bytes for PDF
│   ├── story_engine.py             # Domain detection, narrative assembly
│   ├── pdf_builder.py              # ReportLab PDF with premium fonts
│   ├── bi_engine.py                # Cohort analysis, benchmarking
│   ├── stats_engine.py             # Pearson/Spearman, normality tests
│   ├── ml_engine.py                # Auto model selection, SHAP, CV
│   ├── eda_engine.py               # Distribution analysis, ADF test
│   ├── insight_engine.py           # Insight scoring and ranking
│   ├── insights_builder.py         # Domain-aware insight cards
│   └── domain_dashboards.py        # HR/Sales/Finance/E-com dashboards
│
├── ai/                             # LLM integration
│   ├── llm_client.py               # Groq client with retry logic
│   ├── prompt_builder.py           # Domain-aware prompt templates
│   ├── report_narrator.py          # Chart + executive narrative generation
│   └── tool_dispatcher.py          # AI chat tool routing
│
├── components/
│   └── styles.py                   # Shared CSS — sidebar, typography
│
├── assets/
│   └── fonts/                      # Carlito + Liberation (PDF fonts)
│       ├── Carlito-Regular.ttf
│       ├── Carlito-Bold.ttf
│       └── Liberation-*.ttf
│
├── config/
│   └── settings.py
│
└── tests/                          # 23 unit tests
    ├── test_data_loader.py         # 13 tests — CSV, JSON, dtype inference
    ├── test_domain_detection.py    # 6 tests — HR, Sales, Finance, E-com
    └── test_pdf_builder.py         # 4 tests — build_pdf, themes, regression
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --tb=short -W ignore::DeprecationWarning

# Lint check
ruff check . --select=E,F,W --ignore=E501,E402,F401,W291,W293,E701
```

**Current status:** 23/23 passing · Ruff: 0 errors · 47 files syntax OK

---

## Deploy on Streamlit Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. **New app** → connect GitHub → select `dataforge-ai` → branch `main` → `app.py`
4. **Advanced settings** → Secrets:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```
5. Click **Deploy**

> First deploy takes ~3 minutes to install dependencies.

---

## Freelance Use

DataForge AI is built as a **service delivery tool** — you run it locally, upload the client's dataset, and deliver the generated PDFs.

**Typical workflow:**
```
Client sends CSV → You run DataForge → Deliver 3 files:
  1. cleaned_data.csv        (data cleaning deliverable)
  2. analysis_report.pdf     (17-page branded analysis)
  3. health_report.pdf       (5-page data health assessment)
```

**Suggested pricing:**
| Package | Deliverables | Rate |
|---|---|---|
| Basic | Health report only | $25–40 |
| Standard | Cleaning + Health report | $45–65 |
| Premium | Cleaning + Analysis + Health report | $75–120 |

---

## Roadmap

- [x] Phase 1 — Core platform (upload, quality, dashboard, AI chat)
- [x] Phase 2 — Premium PDF reports with AI narratives
- [x] Phase 3 — ML predictions, SHAP, domain-aware insights
- [x] Phase 4 — CI pipeline, 23 unit tests, lint clean
- [ ] Phase 5 — Multi-file join, saved dashboards, scheduled reports
- [ ] Phase 6 — Database connectors (PostgreSQL, BigQuery, Snowflake)
- [ ] Phase 7 — Client portal — upload link, white-label PDF, payment

---

## Author

**Shweta Mishra**
M.Tech Gold Medalist · ECE · CGPA 9.5 · IFTM University
AI/ML Engineer · Data Analyst · arXiv Researcher

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Shweta_Mishra-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/shweta-mishra-ai)
[![GitHub](https://img.shields.io/badge/GitHub-Shweta--Mishra--ai-181717?style=flat&logo=github)](https://github.com/Shweta-Mishra-ai)

---

## License

© 2026 Shweta Mishra. All rights reserved.

This codebase is proprietary. Unauthorised copying, distribution, modification, or commercial use without explicit written permission is strictly prohibited.

