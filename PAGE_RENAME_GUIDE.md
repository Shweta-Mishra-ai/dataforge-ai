# PAGE_RENAME_GUIDE.md
## How to fix the duplicate sidebar issue in DataForge AI

---

### What was wrong

Streamlit uses filenames to generate sidebar nav entries. When a Python file is
saved on Windows/Mac and pushed to GitHub, emoji characters in filenames can be
re-encoded as hex-escape literals (e.g. `📥` becomes `#L01f4e5`).

This created **two entries per page** in the sidebar:
```
#L01f4e5 Data Upload   ← broken hex-escape duplicate (invisible icon)
📥 Data Upload         ← correct emoji version
```

---

### Fix applied (June 2026)

The following duplicate files were **deleted** from `pages/`:

| Deleted (broken) | Kept (correct) |
|---|---|
| `1_#L01f4e5_Data_Upload.py` | `1_📥_Data_Upload.py` |
| `2_#L01f9f9_Data_Quality.py` | `2_🧹_Data_Quality.py` |
| `3_#L01f4ca_Dashboard.py` | `3_📊_Dashboard.py` |
| `9_#L01f916_AI_Chat.py` | `9_🤖_AI_Chat.py` |
| `10_#L01f52c_Deep_Analysis.py` | `10_🔬_Deep_Analysis.py` |
| `11_#L01f4cb_Health_Report.py` | `11_📋_Health_Report.py` |

The **emoji-named files** are the canonical versions. The hex-escape files
were exact duplicates (or contained older code) and are now removed.

---

### How to prevent this in future

**If you ever rename a page file**, do it via the GitHub web UI or via
Linux/Mac terminal — **not Windows Explorer or Windows Git tools**,
as they can corrupt emoji characters in filenames.

**Safe rename steps (terminal):**
```bash
# Example: rename page 3 from Dashboard to Analytics
cd pages/
mv "3_📊_Dashboard.py" "3_📊_Analytics.py"
git add -A
git commit -m "rename: Dashboard → Analytics"
git push
```

**Safe rename via GitHub web UI:**
1. Open the file on github.com
2. Click the pencil (edit) icon
3. Change the filename at the top
4. Commit changes

---

### Page number ordering

Streamlit renders pages in **filename alphabetical order**. The number prefix
controls the order in the sidebar:

```
1_📥_Data_Upload.py       → appears first
2_🧹_Data_Quality.py      → appears second
3_📊_Dashboard.py          → appears third
4_Business_Insights.py    → appears fourth (no emoji = text nav label)
5_ML_Predictions.py
6_Deep_EDA.py
7_Business_Intel.py
8_Reports.py              → main PDF report generation page
9_🤖_AI_Chat.py
10_🔬_Deep_Analysis.py
11_📋_Health_Report.py    → health report PDF page
```

To reorder pages, just change the number prefix and commit.

---

### Adding a new page

```bash
# Create new page — number it to control sidebar position
touch "pages/12_📈_New_Feature.py"

# Minimal template:
cat > "pages/12_📈_New_Feature.py" << 'TEMPLATE'
import streamlit as st
from core.session_manager import require_data, get_df

st.set_page_config(page_title="New Feature · DataForge AI", layout="wide")
require_data()
df = get_df()

st.title("📈 New Feature")
st.write("Your content here.")
TEMPLATE
```
