# ⚖️ VerdictAI

**Run structured LLM evaluations in minutes. Get an opinionated recommendation — not just a dashboard.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-verdictai.streamlit.app-green?style=for-the-badge)](https://verdictai-eval.streamlit.app/)
[![GitHub Stars](https://img.shields.io/github/stars/anmol1810rs/verdictai?style=for-the-badge)](https://github.com/anmol1810rs/verdictai)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge)](https://python.org)

---

## What is VerdictAI?

VerdictAI is an open-source LLM evaluation engine that helps AI engineers and TPMs make confident model selection decisions.

Upload your prompt dataset, select the models you want to compare, configure your rubric — and get a clear, scored verdict in minutes. No manual scoring. No ambiguous dashboards. Just a direct answer: **which model performs best for your specific use case.**

> "Which LLM should my team use for data annotation QA?" → VerdictAI answers that question with data.

---

## Live Demo

🔗 **[verdictai-eval.streamlit.app](https://verdictai-eval.streamlit.app/)**

> **Note:** The backend runs on Render's free tier and may take 30-60 seconds to wake up on first load. Please be patient on the first request.

**You'll need at least one API key to run evaluations:**
- OpenAI (required — also powers the judge model)
- Anthropic (optional)
- Google Gemini (optional — has a generous free tier)

---

## How It Works

```
1. Upload your prompt dataset (CSV, JSONL, or ZIP for image+text)
         ↓
2. Configure your evaluation rubric
   (choose a preset or set custom dimension weights)
         ↓
3. Select 2-3 models to compare
   (OpenAI, Anthropic, Google — mix and match)
         ↓
4. VerdictAI runs all models in parallel
   and scores every response using GPT-5.4 as judge
         ↓
5. Get a plain-language verdict:
   "GPT-5.4-mini delivers comparable quality 
    at 74% lower cost — recommended for 
    high-volume pipelines."
```

---

## Features

### 📊 Structured Evaluation
- **5 evaluation dimensions:** Factual Accuracy, Hallucination Rate, Instruction Following, Conciseness, Cost Efficiency
- **3 rubric presets:** Customer Support, Technical Documentation, Data Labeling QA
- **Custom weights:** Set your own dimension priorities (hallucination weight enforced minimum)
- **LLM-as-Judge:** GPT-5.4 at temperature=0 scores every response deterministically

### 🔍 Hallucination Detection
- First-class hallucination scoring — not an afterthought
- Models flagged for hallucinations on >30% of prompts are **disqualified from winning** regardless of other scores
- Per-prompt hallucination flags with quoted evidence from the judge

### 💰 Cost Transparency
- Real USD cost per eval run from live pricing
- Cost per 1K tokens and cost per quality point
- Auto-generated cost comparison: "Model A costs $X more but scores Y points higher"
- Prices last updated date shown prominently

### 📈 Prompt Variance Analysis
- Prompts ranked by score variance across models
- ⚡ High variance prompts highlighted — these are where models disagree most
- Click any prompt to see full responses side by side with judge reasoning

### 🖼️ Multimodal Support
- Text prompts (CSV or JSONL)
- Image + text prompts (ZIP with manifest.json)
- Structured data prompts (JSONL with embedded JSON)
- Images displayed inline in results for visual verification

### 📐 Ground Truth Comparison
- Optional expected_output column in your dataset
- Judge alignment score + ROUGE-1 and ROUGE-L scores (objective metrics)
- Verdict explicitly references GT alignment when provided

### 📁 Export
- **PDF report** — 2-page verdict report with scores, reasoning, cost breakdown
- **JSON export** — full structured output with consistent schema for programmatic use

### 🕐 Run History
- All eval runs persisted to SQLite
- Filter by model, engineer name, or date
- Click any past run to reload full results
- Compare any two runs side by side with score deltas

---

## Supported Models (MVP)

| Provider | Model | Tier | Free Tier |
|---|---|---|---|
| OpenAI | GPT-5.4 | Flagship | ❌ |
| OpenAI | GPT-5.4 mini | Budget | ❌ |
| Anthropic | Claude Sonnet 4.6 | Flagship | ❌ (~$5 starter) |
| Anthropic | Claude Haiku 4.5 | Budget | ❌ |
| Google | Gemini 2.5 Pro | Flagship | ✅ 25 req/day |
| Google | Gemini 2.5 Flash | Budget | ✅ 1,000 req/day |

> 💡 **New to VerdictAI?** Use Gemini 2.5 Pro vs Gemini 2.5 Flash for a completely free first eval run.

---

## Quick Start

### Option 1 — Use the live demo
Visit [verdictai-eval.streamlit.app](https://verdictai-eval.streamlit.app/) and bring your own API keys.

### Option 2 — Run locally

```bash
# Clone the repo
git clone https://github.com/anmol1810rs/verdictai.git
cd verdictai

# Set up environment
cp .env.example .env
# Edit .env and set DEV_MODE=true for development

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Run backend
uvicorn backend.main:app --reload --port 8000

# In a new terminal, run frontend
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

### Sample data

Test files are included in the repo:

```
sample_prompts.csv        — 10 text prompts with ground truth
sample_prompts.jsonl      — same data in JSONL format
sample_prompts_no_gt.csv  — text prompts without ground truth
```

---

## Upload Format

### Text prompts (CSV or JSONL)
```csv
prompt,expected_output,engineer_name
"What is machine learning?","Machine learning is...","anmol"
"Explain prompt engineering","Prompt engineering is...","sarah"
```

Required column: `prompt`
Optional columns: `expected_output`, `engineer_name`
Limits: 5–100 prompts per run

### Image + text prompts (ZIP)
```
your_dataset.zip
├── manifest.json
└── images/
    ├── img_001.png
    ├── img_002.jpg
    └── img_003.png
```

```json
{
  "prompts": [
    {
      "id": "img_001",
      "prompt": "Describe what you see in this image",
      "image": "images/img_001.png",
      "expected_output": "Optional ground truth",
      "engineer_name": "anmol"
    }
  ]
}
```

---

## Architecture

```
┌─────────────────────┐         ┌──────────────────────┐
│   Streamlit Frontend │ ──────▶ │   FastAPI Backend     │
│   (Streamlit Cloud)  │         │   (Render free tier)  │
└─────────────────────┘         └──────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
             ┌────────────┐      ┌──────────────┐      ┌─────────────┐
             │  OpenAI    │      │  Anthropic   │      │   Google    │
             │  GPT-5.4   │      │  Claude      │      │   Gemini    │
             └────────────┘      └──────────────┘      └─────────────┘
                                          │
                                          ▼
                                 ┌──────────────┐
                                 │  GPT-5.4     │
                                 │  Judge Model │
                                 │  (temp=0)    │
                                 └──────────────┘
                                          │
                                          ▼
                                 ┌──────────────┐
                                 │  SQLite DB   │
                                 │  (4 tables)  │
                                 └──────────────┘
```

**Stack:**
- Backend: FastAPI + asyncio + SQLAlchemy + SQLite
- Frontend: Streamlit
- Eval: LLM-as-Judge (GPT-5.4, temperature=0)
- Metrics: ROUGE-1, ROUGE-L (objective GT alignment)
- Export: ReportLab (PDF), JSON
- Deployment: Render (backend) + Streamlit Community Cloud (frontend)

---

## Evaluation Methodology

VerdictAI uses **LLM-as-Judge** — a widely adopted evaluation approach where a capable language model scores other models' outputs against a structured rubric.

**Judge model:** GPT-5.4 at temperature=0 (deterministic)
**Judge cost:** Runs on your OpenAI API key — creator pays $0

**Why LLM-as-Judge?**
- Scales to any domain without labeled training data
- Provides reasoning and evidence for every score
- Temperature=0 ensures consistent, reproducible scoring
- Significantly faster and cheaper than human evaluation at scale

**Limitations to be aware of:**
- Self-preference bias possible when judging the same model family
- Judge scores are subjective — ROUGE scores provide objective complement
- Not a replacement for human evaluation in high-stakes production decisions

---

## Cost Structure

**VerdictAI is designed to cost you nothing post-launch.**

| Cost item | Who pays |
|---|---|
| Eval model API calls | User (their own API key) |
| Judge model API calls | User (their OpenAI key) |
| Render backend hosting | Creator (free tier) |
| Streamlit frontend | Creator (free tier) |
| **Total creator cost post-launch** | **$0/month** |

---

## Limitations (MVP)

- **Run history resets** on backend restarts (Render free tier, no persistent disk) — fix coming in v1.1
- **100 prompt maximum** per eval run — increases to 500 in v1.1
- **OpenAI key required** for judge model regardless of which eval models you use
- **Render cold start** — first request after inactivity takes 30-60 seconds
- **No authentication** — single deployment, no user accounts

---

## Roadmap

**v1.1 (next)**
- PostgreSQL migration for persistent run history
- 500 prompt cap
- Historical run comparison with drift detection
- Additional model support (Mistral, Cohere, Ollama)
- Batch API pricing estimates

**v2.0**
- React frontend
- User authentication and team workspaces
- API mode for CI/CD pipeline integration
- Synthetic ground truth generation

---

## Contributing

Contributions welcome. Key areas:

- **Pricing updates** — open a PR updating `pricing.yaml` with current prices and source URL
- **New model support** — add to `models.yaml` and implement provider in `runner.py`
- **Bug fixes** — check open issues

```bash
# Run tests
cd backend
pytest --asyncio-mode=auto -v
# 150 tests, 0 failures
```

---

## Built With

This project was built as a portfolio piece to demonstrate product thinking + engineering execution. The full PRD is included in the repo (`VerdictAI_PRD.docx`).

**Built by [Anmol Malhotra](https://www.linkedin.com/in/anmol-1810rs/)**
AI Engineer → TPM/PM | Vancouver, BC
Currently at Innodata Inc.

---

## License

MIT — use it, fork it, build on it.

---

*VerdictAI — because "it depends" is not a model recommendation.*
