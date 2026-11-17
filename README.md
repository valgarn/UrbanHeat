# City of Dallas: Urban Heat Island Resilience & Cooling Center Allocation Tool
### Agents for Impact 2026 — Challenge 1: Walkability & Urban Heat

An autonomous climate resilience decision support system and interactive web dashboard built with **Google ADK (Agent Development Kit)**, **Gemini 2.5 Flash on Vertex AI**, and **BigQuery ML**.

---

## 🌟 Live Deployments

- **Interactive Decision Support Dashboard**: Runs locally at `http://localhost:8000` or deployable to Vercel via [`web/vercel.json`](web/vercel.json).
- **Cloud Run Agent & Web UI**: Deployed to [https://dallas-heat-planner-447063751013.us-central1.run.app](https://dallas-heat-planner-447063751013.us-central1.run.app)
- **BigQuery ML Model**: `a4i_climate.heat_offset_model` (Linear regression with `ML.EXPLAIN_PREDICT` on Vertex AI `us-central1`).

---

## 🏛️ Executive Summary: Siting Recommendations

Using BigQuery spatial joins (`ST_DISTANCE`), Census tract vulnerability metrics, and BigQuery ML feature attributions across 19,491 street observations and 618 Dallas census tracts, the system identified **5 high-impact sites**:

| Site # | Type | Census Tract | Predicted Offset | Zero-Vehicle % | Poverty % | Population | Primary Decision Driver |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Permanent Center #1 | `48113020500` | +0.05°F | **45.2%** | 29.1% | 6,420 | High Transit Burden (45.2% no-car) |
| **2** | Permanent Center #2 | `48113008704` | -0.56°F | 33.5% | **52.2%** | 4,500 | Concentrated Poverty (52.2%) |
| **3** | Permanent Center #3 | `48113020300` | +0.09°F | **38.5%** | 30.1% | 2,258 | Transit Burden & Multi-Family Density |
| **4** | Permanent Center #4 | `48113019213` | **+0.96°F** | 28.5% | 36.1% | 3,599 | High Modeled Urban Heat Offset (+0.96°F) |
| **5** | Mobile Hydration Unit | `48085031720` | **+1.24°F** | 26.2% | 35.3% | 4,087 | Peak Heat Hazard (+1.24°F) & Senior Vulnerability |

---

## ⚖️ Civil Rights & Equity Audit

The recommended sites underwent an equity verification comparing target tracts against the Dallas citywide baseline:

| Metric | Citywide Baseline | Recommended Siting Cohort | Impact Multiplier |
| :--- | :--- | :--- | :--- |
| **Poverty Rate** | 12.2% | **36.6%** | **3.0x higher** concentration |
| **Zero-Vehicle Households** | 5.3% | **34.4%** | **6.5x higher** transit burden |
| **Population Directly Served** | 2,756,196 | **20,864 residents** | Frontline protection |
| **Median Population Age** | 36.2 yrs | **31.8 yrs** | Younger families with children |

---

## 🌡️ Historical Temperature Exposure (2019–2025)

Analysis of 27,187 Dallas weather station observations:
- **Annual Average Days > 95°F**: **93.4 days**
- **Annual Average Days > 100°F**: **38.0 days**
- **Peak Escalation**: In 2022 and 2023, extreme heat peaked at **65 days > 100°F** each year with maximum temperatures reaching **111.9°F**.

---

## 📂 Repository Structure

- [`agent/agent.py`](agent/agent.py): ADK agent configuration (`dallas_heat_planner`) using Gemini 2.5 Flash on Vertex AI.
- [`agent/tools.py`](agent/tools.py): Custom tools calling BigQuery ML (`explain_heat_offset`, `rank_cooling_center_sites`, `get_heat_frequency`, `run_equity_audit`).
- [`agent/requirements.txt`](agent/requirements.txt): Python dependencies for the agent and tools.
- [`web/static/index.html`](web/static/index.html): Interactive municipal resilience web dashboard with Leaflet map, Chart.js metrics, and embedded AI analyst copilot.
- [`web/server.py`](web/server.py): FastAPI backend providing REST endpoints and streaming chat.
- [`web/vercel.json`](web/vercel.json): Vercel configuration for static/frontend deployment.
- [`test_planner_scenarios.py`](test_planner_scenarios.py): Automated test runner validating all municipal planning workflows.

---

## 🚀 Running Locally

1. **Install dependencies**:
   ```bash
   pip install -r agent/requirements.txt
   ```

2. **Launch Interactive Web Dashboard**:
   ```bash
   python3 web/server.py
   # Opens on http://localhost:8000
   ```

3. **Run Automated Test Scenarios**:
   ```bash
   python3 test_planner_scenarios.py
   ```
