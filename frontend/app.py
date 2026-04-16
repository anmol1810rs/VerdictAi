"""
VerdictAI — Streamlit frontend.
Session 1: layout, API key entry (Story 1.2), file upload (Story 1.1),
           rubric configurator (Story 1.3).
Session 2: full validation (Story 1.6), modality detection (Story 1.5),
           engineer tagging + run label (Story 1.4).
Session 3: background eval + status polling (Story 1.7),
           run history sidebar with filters (Story 1.8).
Session 4: real multi-model runner + LLM-as-Judge + verdict display (Stories 2.1, 2.2).
"""
import os
import time

import requests
import streamlit as st
import yaml

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VerdictAI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load config from YAMLs ─────────────────────────────────────────────────

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")

@st.cache_resource
def load_models_config() -> dict:
    with open(os.path.join(REPO_ROOT, "models.yaml")) as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_pricing_config() -> dict:
    with open(os.path.join(REPO_ROOT, "pricing.yaml")) as f:
        return yaml.safe_load(f)

models_cfg = load_models_config()
pricing_cfg = load_pricing_config()

MVP_MODELS = models_cfg["mvp_models"]
PRESETS = models_cfg["rubric_presets"]

# ── Modality display helpers ───────────────────────────────────────────────

MODALITY_ICONS = {
    "text": "📄",
    "image_text": "🖼️",
    "structured_data": "📊",
}

STATUS_ICONS = {
    "complete": "✅",
    "running": "⏳",
    "failed": "❌",
    "pending": "🔄",
}

# ── Session state defaults ─────────────────────────────────────────────────

if "uploaded_prompts" not in st.session_state:
    st.session_state.uploaded_prompts = None
if "upload_meta" not in st.session_state:
    st.session_state.upload_meta = None
if "detected_modality" not in st.session_state:
    st.session_state.detected_modality = "text"
if "rubric" not in st.session_state:
    st.session_state.rubric = None
if "last_run_id" not in st.session_state:
    st.session_state.last_run_id = None
if "polling_run_id" not in st.session_state:
    st.session_state.polling_run_id = None  # run_id being actively polled
if "eval_complete_banner" not in st.session_state:
    st.session_state.eval_complete_banner = False  # show completion notice until user dismisses
if "history_model_filter" not in st.session_state:
    st.session_state.history_model_filter = ""
if "history_engineer_filter" not in st.session_state:
    st.session_state.history_engineer_filter = ""


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR — API Keys (Story 1.2) + Run History (Story 1.8)
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚖️ VerdictAI")
    st.caption("Run structured LLM evaluations in minutes.")
    st.divider()

    # ── API Keys ──────────────────────────────────────────────────────────
    st.header("🔑 API Keys")
    st.caption(
        "Keys are used only for this session and **never stored**. "
        "Your keys pay for all inference costs including the judge model."
    )

    openai_key = st.text_input(
        "OpenAI API Key *",
        type="password",
        placeholder="sk-...",
        help="Required — used for eval model calls AND the judge model.",
        key="openai_key_input",
    )
    anthropic_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Required only if you select an Anthropic model.",
        key="anthropic_key_input",
    )
    google_key = st.text_input(
        "Google API Key",
        type="password",
        placeholder="AIza...",
        help="Required only if you select a Google Gemini model. Free tier available.",
        key="google_key_input",
    )

    if openai_key:
        if openai_key.startswith("sk-") and len(openai_key) >= 20:
            st.success("OpenAI key provided ✓")
        else:
            st.error("Invalid OpenAI key format — must start with 'sk-'")
    else:
        st.warning("OpenAI key required to run evaluations.")

    st.caption("💡 Google Gemini has a **free tier** — 1,000 requests/day, no credit card required.")

    # ── Run History (Story 1.8) ───────────────────────────────────────────
    st.divider()
    st.header("📋 Run History")

    # Filter bar
    with st.expander("🔍 Filters", expanded=False):
        filter_model = st.text_input(
            "Filter by model",
            value=st.session_state.history_model_filter,
            placeholder="e.g. gpt-5-4",
            key="sidebar_model_filter",
        )
        filter_engineer = st.text_input(
            "Filter by engineer",
            value=st.session_state.history_engineer_filter,
            placeholder="e.g. Alice",
            key="sidebar_engineer_filter",
        )
        col_from, col_to = st.columns(2)
        with col_from:
            filter_date_from = st.date_input("From", value=None, key="sidebar_date_from")
        with col_to:
            filter_date_to = st.date_input("To", value=None, key="sidebar_date_to")

        if st.button("Clear filters", key="clear_filters"):
            st.session_state.history_model_filter = ""
            st.session_state.history_engineer_filter = ""
            st.rerun()

    # Fetch history from backend
    try:
        params: dict = {}
        if filter_model:
            params["model"] = filter_model
        if filter_engineer:
            params["engineer"] = filter_engineer
        if filter_date_from:
            params["date_from"] = filter_date_from.isoformat()
        if filter_date_to:
            params["date_to"] = filter_date_to.isoformat()

        hist_resp = requests.get(f"{BACKEND_URL}/eval/history", params=params, timeout=5)
        if hist_resp.status_code == 200:
            history_data = hist_resp.json()
            runs = history_data.get("runs", [])

            if not runs:
                st.caption("No eval runs yet. Upload a dataset to get started.")
            else:
                for run in runs:
                    modality_icon = MODALITY_ICONS.get(run["modality"], "📄")
                    status_icon = STATUS_ICONS.get(run["status"], "❓")
                    label = run.get("run_label") or "Unnamed run"
                    models_str = ", ".join(run.get("models_selected", []))

                    # Build button label
                    btn_label = f"{status_icon} {label}"
                    btn_help = (
                        f"{modality_icon} {run['modality']} | "
                        f"{run['created_at']} | "
                        f"Models: {models_str}"
                    )
                    if run["status"] == "complete" and run.get("winning_model"):
                        btn_help += f" | Winner: {run['winning_model']}"
                    if run["status"] == "failed" and run.get("error_message"):
                        btn_help += f" | Error: {run['error_message'][:60]}..."

                    if st.button(btn_label, key=f"hist_{run['id']}", help=btn_help, use_container_width=True):
                        if run["status"] == "complete":
                            st.session_state.last_run_id = run["id"]
                            st.session_state.polling_run_id = None
                            st.success("Results loaded. Switch to **Results** tab.")
                        else:
                            st.info(f"Run status: {run['status']}")
        else:
            st.caption("Could not load run history.")
    except requests.exceptions.ConnectionError:
        st.caption("Backend offline — history unavailable.")


# ══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT — Tabs
# ══════════════════════════════════════════════════════════════════════════

st.title("⚖️ VerdictAI")
st.caption("Open-source LLM evaluation engine. Run structured evals, get an opinionated verdict.")

tab_upload, tab_rubric, tab_models, tab_run, tab_results = st.tabs([
    "📁 Upload Dataset",
    "🎛️ Configure Rubric",
    "🤖 Select Models",
    "▶️ Run Evaluation",
    "📊 Results",
])


# ──────────────────────────────────────────────────────────────────────────
# TAB 1 — Upload Dataset (Story 1.1 + 1.6)
# ──────────────────────────────────────────────────────────────────────────

with tab_upload:
    st.header("Upload Dataset")

    st.info(
        "**Supported formats:**\n"
        "- **CSV / JSONL** — Text prompts. Required column: `prompt`. "
        "Optional: `expected_output`, `engineer_name`.\n"
        "- **ZIP** — Image + text. Must contain `manifest.json` and an `images/` folder.\n\n"
        f"**Limits:** {5}–100 prompts per run (MVP)."
    )

    with st.expander("📥 Download upload templates"):
        csv_template = "prompt,expected_output,engineer_name\n" \
                       "What is the capital of France?,Paris,Alice\n" \
                       "Summarise this article in one sentence.,,Bob\n"
        st.download_button(
            "Download CSV template",
            data=csv_template,
            file_name="verdictai_template.csv",
            mime="text/csv",
        )
        jsonl_template = '{"prompt": "What is the capital of France?", "expected_output": "Paris", "engineer_name": "Alice"}\n' \
                         '{"prompt": "Summarise this article in one sentence."}\n'
        st.download_button(
            "Download JSONL template",
            data=jsonl_template,
            file_name="verdictai_template.jsonl",
            mime="application/json",
        )

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "jsonl", "zip"],
        help="CSV or JSONL for text prompts. ZIP for image+text datasets.",
    )

    if uploaded_file is not None:
        with st.spinner("Parsing and validating dataset..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    timeout=10,
                )
                if response.status_code == 200:
                    meta = response.json()
                    st.session_state.upload_meta = meta
                    st.session_state.uploaded_prompts = meta["prompts"]
                    st.session_state.detected_modality = meta.get("modality", "text")

                    st.success(f"✓ {meta.get('validation_summary', 'Dataset validated')}")

                    if meta.get("warnings"):
                        with st.expander("⚠️ Warnings (non-blocking)"):
                            for warn in meta["warnings"]:
                                st.warning(f"**{warn['field']}:** {warn['message']}")

                else:
                    detail = response.json().get("detail", "Unknown error")
                    st.error(f"❌ Validation failed: {detail}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach backend. Make sure the FastAPI server is running on port 8000.")

    if st.session_state.upload_meta:
        with st.expander("Preview parsed prompts"):
            prompts = st.session_state.uploaded_prompts or []
            for i, p in enumerate(prompts[:5], 1):
                st.markdown(f"**{i}.** {p['prompt']}")
            if len(prompts) > 5:
                st.caption(f"... and {len(prompts) - 5} more prompts")


# ──────────────────────────────────────────────────────────────────────────
# TAB 2 — Rubric Configurator (Story 1.3)
# ──────────────────────────────────────────────────────────────────────────

with tab_rubric:
    st.header("Configure Evaluation Rubric")
    st.caption(
        "Define what 'good' looks like for your use case. "
        "Weights must sum to 100. Hallucination cannot be zero."
    )

    col_preset, col_info = st.columns([2, 3])
    with col_preset:
        preset_options = {
            "Custom": None,
            "💬 Customer Support": "customer_support",
            "📄 Technical Documentation": "technical_documentation",
            "🏷️ Data Labeling QA": "data_labeling_qa",
        }
        selected_label = st.selectbox("Start from a preset", list(preset_options.keys()))
        selected_preset_key = preset_options[selected_label]

    if selected_preset_key:
        preset_weights = PRESETS[selected_preset_key]["weights"]
        with col_info:
            st.info(f"**{PRESETS[selected_preset_key]['display_name']}** — {PRESETS[selected_preset_key]['description']}")
    else:
        preset_weights = {"accuracy": 20, "hallucination": 20, "instruction_following": 20, "conciseness": 20, "cost_efficiency": 20}
        with col_info:
            st.info("Set custom weights below. All five must sum to exactly 100.")

    st.divider()
    st.subheader("Dimension Weights")

    col1, col2 = st.columns(2)
    with col1:
        w_accuracy = st.slider("Factual Accuracy", 0, 100, preset_weights.get("accuracy", 20),
                               help="Does the response contain correct, verifiable information?")
        w_hallucination = st.slider("Hallucination Rate ⚠️", 10, 100, max(preset_weights.get("hallucination", 20), 10),
                                    help="Minimum 10 — mandatory. High hallucination overrides positive verdict.")
        w_instruction = st.slider("Instruction Following", 0, 100, preset_weights.get("instruction_following", 20),
                                  help="Did the model do exactly what was asked?")
    with col2:
        w_conciseness = st.slider("Conciseness", 0, 100, preset_weights.get("conciseness", 20),
                                  help="Is the response appropriately sized — not too long, not too short?")
        w_cost = st.slider("Cost Efficiency", 0, 100, preset_weights.get("cost_efficiency", 20),
                           help="Cost per quality point — calculated automatically from token usage.")

    total = w_accuracy + w_hallucination + w_instruction + w_conciseness + w_cost
    st.metric("Total weight", f"{total} / 100", delta=total - 100)

    if total == 100:
        rubric = {
            "accuracy": w_accuracy,
            "hallucination": w_hallucination,
            "instruction_following": w_instruction,
            "conciseness": w_conciseness,
            "cost_efficiency": w_cost,
        }
        try:
            val = requests.post(f"{BACKEND_URL}/rubric/validate", json=rubric, timeout=5)
            if val.status_code == 200:
                st.session_state.rubric = rubric
                st.success("Rubric valid ✓")
            else:
                st.error(f"Rubric rejected: {val.json()}")
        except requests.exceptions.ConnectionError:
            st.session_state.rubric = rubric
    else:
        st.warning(f"Weights must sum to 100. Current: {total}. Adjust the sliders.")
        st.session_state.rubric = None


# ──────────────────────────────────────────────────────────────────────────
# TAB 3 — Model Selection (Story 1.5)
# ──────────────────────────────────────────────────────────────────────────

with tab_models:
    st.header("Select Models to Compare")
    st.caption("Compatible models shown based on your dataset's modality.")

    detected_modality = st.session_state.get("detected_modality", "text")

    if st.session_state.upload_meta:
        st.info(f"📊 **Detected modality:** {detected_modality}")

        with st.spinner("Loading compatible models..."):
            try:
                compat_resp = requests.get(
                    f"{BACKEND_URL}/models/compatible",
                    params={"modality": detected_modality},
                    timeout=10,
                )
                if compat_resp.status_code == 200:
                    compat_data = compat_resp.json()
                    incompatible_list = compat_data.get("incompatible_models", [])
                    suggestions = compat_data.get("suggestions", {})

                    st.subheader("✅ Compatible Models")
                    selected_models = []
                    cols = st.columns(2)
                    for i, model in enumerate(compat_data["compatible_models"]):
                        with cols[i % 2]:
                            checked = st.checkbox(
                                f"🤖 **{model['display_name']}**",
                                key=f"model_{model['id']}",
                                help=f"{model.get('description', '')} ({model['provider']})",
                            )
                            if checked:
                                selected_models.append(model["id"])

                    if incompatible_list:
                        st.subheader("❌ Incompatible Models")
                        for inc in incompatible_list:
                            suggestion = suggestions.get(inc["model"], {})
                            if suggestion:
                                st.warning(f"**{inc['model']}**: {inc['reason']} → "
                                          f"Switch to **{suggestion.get('suggest')}**?")
                            else:
                                st.info(f"**{inc['model']}**: {inc['reason']}")

                    if selected_models:
                        if len(selected_models) > 3:
                            st.warning("Select at most 3 models for comparison.")
                        else:
                            st.success(f"Selected: {', '.join(selected_models)}")
                        st.session_state.selected_models = selected_models
                    else:
                        st.info("Select 2–3 models to continue.")

                else:
                    st.error(f"Error loading models: {compat_resp.json()}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach backend to fetch compatible models.")
    else:
        st.info("Upload a dataset first to see compatible models.")

    st.divider()
    st.caption(
        "💡 **Free tier available:** Gemini 2.5 Flash offers 1,000 requests/day at no cost. "
        "Best entry point for testing VerdictAI without spending on API credits."
    )


# ──────────────────────────────────────────────────────────────────────────
# TAB 4 — Run Evaluation (Story 1.7: background eval + status polling)
# ──────────────────────────────────────────────────────────────────────────

with tab_run:
    st.header("Run Evaluation")

    models = st.session_state.get("selected_models", [])
    prompts = st.session_state.uploaded_prompts
    rubric = st.session_state.rubric

    checks = {
        "Dataset uploaded": prompts is not None,
        "Rubric configured": rubric is not None,
        "Models selected (2–3)": 2 <= len(models) <= 3,
        "OpenAI key provided": bool(openai_key and openai_key.startswith("sk-")),
    }
    for label, passed in checks.items():
        icon = "✅" if passed else "❌"
        st.markdown(f"{icon} {label}")

    st.divider()

    engineer_name = st.text_input(
        "Prompt engineer name (optional)",
        placeholder="e.g. Alice",
        help="Tags ALL prompts in this batch to a team member. (Optional)",
    )

    run_label = st.text_input(
        "Run label (optional)",
        placeholder="e.g. Sprint 4 QA or April batch 1",
        help="Add a descriptive label to identify this evaluation run later.",
        key="run_label_input",
    )

    all_ready = all(checks.values())

    # ── Story 1.7: status polling loop ────────────────────────────────────
    polling_run_id = st.session_state.get("polling_run_id")
    if polling_run_id:
        try:
            status_resp = requests.get(f"{BACKEND_URL}/eval/{polling_run_id}/status", timeout=5)
            if status_resp.status_code == 200:
                status_data = status_resp.json()
                current_status = status_data["status"]

                if current_status == "pending":
                    st.info("🔄 Run queued...")
                    time.sleep(2)
                    st.rerun()
                elif current_status == "running":
                    st.info("⏳ Evaluation in progress... (refreshing every 2s)")
                    time.sleep(2)
                    st.rerun()
                elif current_status == "complete":
                    st.session_state.last_run_id = polling_run_id
                    st.session_state.polling_run_id = None
                    st.session_state.eval_complete_banner = True
                    st.rerun()
                elif current_status == "failed":
                    error_msg = status_data.get("error_message", "Unknown error")
                    st.session_state.polling_run_id = None
                    st.error(f"❌ Evaluation failed: {error_msg}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach backend. Is the FastAPI server running?")

    # Persistent completion banner — stays until user starts a new run
    if st.session_state.eval_complete_banner:
        st.success(
            "✅ **Evaluation complete!** Switch to the **📊 Results** tab to view scores and verdict.",
            icon="🎉",
        )
        if st.button("Clear notification", key="clear_banner"):
            st.session_state.eval_complete_banner = False
            st.rerun()

    if st.button("▶️ Start Evaluation", disabled=not all_ready, type="primary"):
        api_keys = {"openai_api_key": openai_key}
        if anthropic_key:
            api_keys["anthropic_api_key"] = anthropic_key
        if google_key:
            api_keys["google_api_key"] = google_key

        payload = {
            "prompts": prompts,
            "models_selected": models,
            "rubric": rubric,
            "api_keys": api_keys,
            "engineer_name": engineer_name or None,
            "custom_label": run_label or None,
        }

        try:
            resp = requests.post(f"{BACKEND_URL}/eval/run", json=payload, timeout=30)
            if resp.status_code == 200:
                run_id = resp.json()["run_id"]
                st.session_state.polling_run_id = run_id
                st.session_state.eval_complete_banner = False  # clear previous completion notice
                st.rerun()
            else:
                try:
                    error_detail = resp.json()
                except Exception:
                    error_detail = resp.text or f"HTTP {resp.status_code} (empty response)"
                st.error(f"Eval failed to start: {error_detail}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach backend. Is the FastAPI server running?")

    if not all_ready:
        st.caption("Complete the steps above to enable the run button.")


# ──────────────────────────────────────────────────────────────────────────
# TAB 5 — Results (Session 4: verdict card, score table, cost breakdown, per-prompt)
# ──────────────────────────────────────────────────────────────────────────

# Dimension display names map
DIMENSION_DISPLAY = {
    "accuracy": "Accuracy",
    "hallucination": "Hallucination",
    "instruction_following": "Instruction",
    "conciseness": "Conciseness",
}

DIMENSION_COLORS = {
    "accuracy": "#10a37f",
    "hallucination": "#e53e3e",
    "instruction_following": "#3182ce",
    "conciseness": "#d69e2e",
}


def _score_color(score) -> str:
    """Return green / amber / red CSS color based on 0-10 score."""
    if score is None:
        return "#888888"
    if score >= 7:
        return "#2e7d32"   # green
    if score >= 4:
        return "#e65100"   # amber
    return "#c62828"       # red


def _score_badge(score) -> str:
    """Format score as colored badge string."""
    if score is None:
        return "N/A"
    return f"{score:.1f}"


def _delta_color(delta) -> str:
    """Return green / red / grey CSS color for a score delta."""
    if delta is None:
        return "#888888"
    if delta > 0:
        return "#2e7d32"   # green — improved
    if delta < 0:
        return "#c62828"   # red — regressed
    return "#888888"       # grey — unchanged


def _delta_badge(delta) -> str:
    """Format delta as signed string, e.g. +1.2 or -0.5."""
    if delta is None:
        return "N/A"
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.2f}"


def _render_compare(compare_data: dict) -> None:
    """Render the side-by-side run comparison view."""
    run_a = compare_data["run_a"]
    run_b = compare_data["run_b"]
    deltas = compare_data["deltas"]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Run A:** {run_a['label']}")
        st.caption(f"{run_a['date']} | Models: {', '.join(run_a['models'])}")
        winner_a = run_a.get("winner") or "—"
        st.markdown(f"🏆 Winner: **{winner_a}**")
    with col_b:
        st.markdown(f"**Run B:** {run_b['label']}")
        st.caption(f"{run_b['date']} | Models: {', '.join(run_b['models'])}")
        winner_b = run_b.get("winner") or "—"
        st.markdown(f"🏆 Winner: **{winner_b}**")

    if deltas.get("winner_changed"):
        st.warning("⚡ Winner changed between runs.")

    # Score delta table
    st.markdown("**Score Deltas** (Run B − Run A, per model per dimension)")
    score_delta = deltas.get("score_delta", {})
    cost_delta = deltas.get("cost_delta", {})
    all_compare_models = sorted(score_delta.keys())
    compare_dims = ["accuracy", "hallucination", "instruction_following", "conciseness"]
    dim_display = {
        "accuracy": "Accuracy", "hallucination": "Hallucination",
        "instruction_following": "Instruction", "conciseness": "Conciseness",
    }

    delta_rows = []
    for m in all_compare_models:
        row = {"Model": m}
        for d in compare_dims:
            v = score_delta.get(m, {}).get(d)
            row[dim_display[d]] = _delta_badge(v)
        cd = cost_delta.get(m)
        row["Cost Δ (USD)"] = _delta_badge(cd) if cd is not None else "N/A"
        delta_rows.append(row)

    if delta_rows:
        import pandas as pd
        st.dataframe(pd.DataFrame(delta_rows), use_container_width=True, hide_index=True)

    insight = deltas.get("insight", "")
    if insight:
        st.info(f"💡 {insight}")


with tab_results:
    st.header("📊 Results")

    run_id = st.session_state.get("last_run_id")
    if not run_id:
        st.info("No evaluation run yet. Complete an eval in the **▶️ Run Evaluation** tab.")
    else:
        try:
            resp = requests.get(f"{BACKEND_URL}/eval/{run_id}/results", timeout=10)
            if resp.status_code != 200:
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text
                st.error(f"Could not load results: {err}")
            else:
                data = resp.json()
                results = data.get("results", [])
                verdict = data.get("verdict")

                # ── 1. Verdict Card ────────────────────────────────────────────
                if verdict:
                    winning_model = verdict.get("winning_model", "Unknown")
                    summary = verdict.get("summary", "")
                    warnings = verdict.get("hallucination_warnings", [])

                    st.markdown(
                        f"""
<div style="background:#0d1f17;border:1px solid #10a37f;border-left:6px solid #10a37f;padding:20px;border-radius:8px;margin-bottom:16px">
  <h2 style="color:#10a37f;margin:0">⚖️ Verdict: {winning_model}</h2>
  <p style="margin:8px 0 0 0;white-space:pre-line;color:#d1fae5;line-height:1.6">{summary}</p>
</div>
""",
                        unsafe_allow_html=True,
                    )

                    if warnings:
                        for w in warnings:
                            st.warning(f"⚠️ {w}")
                else:
                    st.info("Verdict not yet generated.")

                if not results:
                    st.info("No model results found for this run.")
                else:
                    # Collect unique models and dimensions
                    all_models = sorted({r["model_name"] for r in results})
                    all_dims = ["accuracy", "hallucination", "instruction_following", "conciseness"]

                    # ── 2. Score Breakdown Table ───────────────────────────────
                    st.subheader("📋 Score Breakdown")
                    st.caption("Average score per model per dimension. Click any model to see per-prompt reasoning.")

                    # Aggregate: average per model per dimension
                    from collections import defaultdict
                    dim_sum: dict = defaultdict(lambda: defaultdict(float))
                    dim_cnt: dict = defaultdict(lambda: defaultdict(int))
                    for r in results:
                        m = r["model_name"]
                        for d in all_dims:
                            v = r.get("dimension_scores", {}).get(d)
                            if v is not None:
                                dim_sum[m][d] += float(v)
                                dim_cnt[m][d] += 1

                    # GT Alignment: aggregate avg per model when GT scores present
                    has_gt_data = any(
                        r.get("ground_truth_score") is not None for r in results
                    )
                    gt_sum: dict = defaultdict(float)
                    gt_cnt: dict = defaultdict(int)
                    if has_gt_data:
                        for r in results:
                            gts = r.get("ground_truth_score")
                            if gts is not None:
                                gt_sum[r["model_name"]] += float(gts)
                                gt_cnt[r["model_name"]] += 1

                    # Build table rows
                    table_rows = []
                    for m in all_models:
                        row = {"Model": m}
                        total = 0.0
                        count = 0
                        for d in all_dims:
                            avg = (
                                dim_sum[m][d] / dim_cnt[m][d]
                                if dim_cnt[m][d] > 0
                                else None
                            )
                            row[DIMENSION_DISPLAY.get(d, d)] = _score_badge(avg)
                            if avg is not None:
                                total += avg
                                count += 1
                        row["Total"] = _score_badge(total / count if count else None)
                        if has_gt_data:
                            gt_avg = gt_sum[m] / gt_cnt[m] if gt_cnt[m] > 0 else None
                            row["GT Alignment"] = _score_badge(gt_avg)
                        is_winner = verdict and m == verdict.get("winning_model")
                        row[""] = "🏆 Winner" if is_winner else ""
                        table_rows.append(row)

                    if table_rows:
                        import pandas as pd
                        df = pd.DataFrame(table_rows)
                        st.dataframe(df, use_container_width=True, hide_index=True)

                    # Inline reasoning viewer
                    with st.expander("🔍 View reasoning per model", expanded=False):
                        selected_model_view = st.selectbox(
                            "Select model", all_models, key="reasoning_model_select"
                        )
                        model_results_for_view = [
                            r for r in results if r["model_name"] == selected_model_view
                        ]
                        for r in model_results_for_view[:5]:
                            st.markdown(f"**Prompt {r['prompt_index']}**")
                            scores = r.get("dimension_scores", {})
                            reasoning = r.get("dimension_reasoning", {})
                            for d in all_dims:
                                score_val = scores.get(d)
                                reason_text = reasoning.get(d, "—")
                                color = _score_color(score_val)
                                st.markdown(
                                    f"<span style='color:{color};font-weight:bold'>"
                                    f"{DIMENSION_DISPLAY.get(d, d)}: {_score_badge(score_val)}/10</span>"
                                    f" — {reason_text}",
                                    unsafe_allow_html=True,
                                )
                            if r.get("hallucination_flagged"):
                                st.error(
                                    f"⚠️ Hallucination flagged: {r.get('hallucination_reason', 'detected')}"
                                )
                            st.divider()

                    # ── 3. Cost Breakdown ──────────────────────────────────────
                    st.subheader("💰 Cost Breakdown")

                    cost_by_model: dict = defaultdict(float)
                    tokens_in_by_model: dict = defaultdict(int)
                    tokens_out_by_model: dict = defaultdict(int)
                    prompt_count_by_model: dict = defaultdict(int)
                    for r in results:
                        m = r["model_name"]
                        cost_by_model[m] += r.get("cost_usd", 0.0)
                        # Prefer direct columns; fall back to tokens_used dict
                        tin_r = r.get("tokens_in")
                        tout_r = r.get("tokens_out")
                        if tin_r is None:
                            tu = r.get("tokens_used", {})
                            tin_r = tu.get("input", 0)
                            tout_r = tu.get("output", 0)
                        tokens_in_by_model[m] += tin_r or 0
                        tokens_out_by_model[m] += tout_r or 0
                        prompt_count_by_model[m] += 1

                    cost_rows = []
                    for m in all_models:
                        total_cost = cost_by_model[m]
                        tin = tokens_in_by_model[m]
                        tout = tokens_out_by_model[m]
                        total_tokens = tin + tout
                        cost_per_1k = (total_cost / total_tokens * 1000) if total_tokens > 0 else 0.0
                        quality_score = (
                            dim_sum[m]["accuracy"] / dim_cnt[m]["accuracy"]
                            if dim_cnt[m]["accuracy"] > 0 else 1.0
                        )
                        cpp = total_cost / quality_score if quality_score > 0 else 0.0
                        cost_rows.append(
                            {
                                "Model": m,
                                "Total Cost (USD)": f"${total_cost:.6f}",
                                "Tokens In": f"{tin:,}",
                                "Tokens Out": f"{tout:,}",
                                "Cost / 1K tokens": f"${cost_per_1k:.6f}",
                                "Cost / Quality Pt": f"${cpp:.6f}",
                            }
                        )

                    if cost_rows:
                        import pandas as pd
                        cost_df = pd.DataFrame(cost_rows)
                        st.dataframe(cost_df, use_container_width=True, hide_index=True)

                    # "Prices last updated" date from pricing.yaml meta
                    _last_updated = pricing_cfg.get("meta", {}).get("last_updated", "")
                    if _last_updated:
                        st.caption(f"Prices last updated: {_last_updated}")

                    # Cost comparison callout from verdict
                    _callout = (verdict or {}).get("cost_comparison", {}).get("callout", "")
                    if _callout:
                        st.info(f"💡 {_callout}")

                    # ── 4. Per-Prompt Breakdown ────────────────────────────────
                    st.subheader("🔎 Per-Prompt Breakdown")
                    st.caption("Expand any prompt to see full model responses and scores.")

                    # Group results by prompt index
                    prompts_map: dict = defaultdict(list)
                    for r in results:
                        prompts_map[r["prompt_index"]].append(r)

                    # Sort by variance_score from API (saved to DB); fall back to client-side calc
                    def _prompt_variance_fallback(prompt_results) -> float:
                        totals = []
                        for r in prompt_results:
                            s = r.get("dimension_scores", {})
                            vals = [v for v in s.values() if v is not None]
                            if vals:
                                totals.append(sum(vals) / len(vals))
                        if len(totals) < 2:
                            return 0.0
                        return max(totals) - min(totals)

                    def _get_variance(idx) -> float:
                        pr = prompts_map[idx]
                        vs = pr[0].get("variance_score")
                        if vs is not None:
                            return vs
                        return _prompt_variance_fallback(pr)

                    sorted_indices = sorted(
                        prompts_map.keys(), key=_get_variance, reverse=True
                    )

                    top_variance_indices = set(sorted_indices[:3]) if len(sorted_indices) > 1 else set()

                    def _variance_insight(prompt_results) -> str:
                        """One-sentence insight: dimension with highest delta across models."""
                        dims = ["accuracy", "hallucination", "instruction_following", "conciseness"]
                        insight_phrases = {
                            "accuracy": "this prompt tests factual knowledge where model training data may differ",
                            "hallucination": "models vary in tendency to fabricate information on this topic",
                            "instruction_following": "models interpreted the instructions differently",
                            "conciseness": "models chose very different response lengths for this prompt",
                        }
                        max_delta, max_dim = 0.0, None
                        for d in dims:
                            vals = [r.get("dimension_scores", {}).get(d) for r in prompt_results]
                            vals = [v for v in vals if v is not None]
                            if len(vals) >= 2:
                                delta = max(vals) - min(vals)
                                if delta > max_delta:
                                    max_delta, max_dim = delta, d
                        if max_dim is None:
                            return ""
                        phrase = insight_phrases.get(max_dim, "models showed different performance")
                        return f"Models differed most on **{max_dim.replace('_', ' ')}** (δ={max_delta:.1f}) — {phrase}."

                    for idx in sorted_indices:
                        prompt_results = prompts_map[idx]
                        first = prompt_results[0]
                        prompt_preview = (first.get("prompt_text") or f"Prompt {idx}")[:80]
                        if len(first.get("prompt_text") or "") > 80:
                            prompt_preview += "..."

                        is_high_variance = idx in top_variance_indices
                        badge = "⚡ " if is_high_variance else ""
                        label = f"{badge}Prompt {idx}: {prompt_preview}"

                        with st.expander(label, expanded=False):
                            if is_high_variance:
                                insight = _variance_insight(prompt_results)
                                st.info(
                                    "⚡ **Models disagreed significantly on this prompt.**"
                                    + (f"\n\n{insight}" if insight else "")
                                )

                            for r in prompt_results:
                                model_name = r["model_name"]
                                scores = r.get("dimension_scores", {})
                                avg_score = None
                                vals = [v for v in scores.values() if v is not None]
                                if vals:
                                    avg_score = sum(vals) / len(vals)

                                badge_color = _score_color(avg_score)
                                st.markdown(
                                    f"<b style='color:{badge_color}'>{model_name}"
                                    f" (avg: {_score_badge(avg_score)}/10)</b>",
                                    unsafe_allow_html=True,
                                )

                                if r.get("hallucination_flagged"):
                                    st.error("⚠️ Hallucination flagged on this prompt.")

                                with st.container():
                                    col_resp, col_scores = st.columns([3, 2])
                                    with col_resp:
                                        st.markdown("**Response:**")
                                        st.text(r.get("response_text", "")[:500])
                                    with col_scores:
                                        st.markdown("**Scores:**")
                                        for d in all_dims:
                                            v = scores.get(d)
                                            st.markdown(
                                                f"<span style='color:{_score_color(v)}'>"
                                                f"{DIMENSION_DISPLAY.get(d, d)}: {_score_badge(v)}</span>",
                                                unsafe_allow_html=True,
                                            )
                                        gt_s = r.get("ground_truth_score")
                                        gt_r = r.get("ground_truth_reasoning", "")
                                        if gt_s is not None:
                                            st.markdown(
                                                f"<span style='color:{_score_color(gt_s)}'>"
                                                f"GT Alignment: {gt_s:.1f}/10</span>",
                                                unsafe_allow_html=True,
                                            )
                                            if gt_r:
                                                st.caption(gt_r)
                                st.divider()

        except requests.exceptions.ConnectionError:
            st.error("Cannot reach backend. Is the FastAPI server running?")

    # ── 5. Export (Story 3.1 & 3.2) ──────────────────────────────────────
    st.divider()
    st.subheader("📤 Export Results")

    run_id_for_export = st.session_state.get("last_run_id")
    _export_complete  = False

    # Check if the current run is complete before enabling export
    if run_id_for_export:
        try:
            _st_resp = requests.get(
                f"{BACKEND_URL}/eval/{run_id_for_export}/status", timeout=5
            )
            if _st_resp.status_code == 200:
                _export_complete = _st_resp.json().get("status") == "complete"
        except requests.exceptions.ConnectionError:
            pass

    _exp_col_pdf, _exp_col_json, _exp_col_clip = st.columns([1, 1, 2])

    with _exp_col_pdf:
        if st.button(
            "📄 Export PDF",
            disabled=not _export_complete,
            key="btn_export_pdf",
            help="Download a 2-page PDF report for this evaluation run.",
        ):
            with st.spinner("Generating PDF…"):
                try:
                    _pdf_resp = requests.get(
                        f"{BACKEND_URL}/eval/{run_id_for_export}/export/pdf",
                        timeout=30,
                    )
                    if _pdf_resp.status_code == 200:
                        st.download_button(
                            label="📄 Download PDF",
                            data=_pdf_resp.content,
                            file_name=f"verdictai_{run_id_for_export[:8]}.pdf",
                            mime="application/pdf",
                            key="dl_pdf",
                        )
                    else:
                        try:
                            _err = _pdf_resp.json().get("detail", _pdf_resp.text)
                        except Exception:
                            _err = _pdf_resp.text
                        st.error(f"PDF export failed: {_err}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach backend.")

    with _exp_col_json:
        if st.button(
            "📋 Export JSON",
            disabled=not _export_complete,
            key="btn_export_json",
            help="Download the full structured JSON export for this evaluation run.",
        ):
            with st.spinner("Building JSON…"):
                try:
                    _json_resp = requests.get(
                        f"{BACKEND_URL}/eval/{run_id_for_export}/export/json",
                        timeout=30,
                    )
                    if _json_resp.status_code == 200:
                        _cd = _json_resp.headers.get("content-disposition", "")
                        import re as _re
                        _fname_match = _re.search(r'filename="([^"]+)"', _cd)
                        _json_fname = (
                            _fname_match.group(1) if _fname_match
                            else f"verdictai_{run_id_for_export[:8]}.json"
                        )
                        st.download_button(
                            label="📋 Download JSON",
                            data=_json_resp.content,
                            file_name=_json_fname,
                            mime="application/json",
                            key="dl_json",
                        )
                        # Store JSON in session for clipboard copy
                        st.session_state["export_json_text"] = _json_resp.text
                    else:
                        try:
                            _err = _json_resp.json().get("detail", _json_resp.text)
                        except Exception:
                            _err = _json_resp.text
                        st.error(f"JSON export failed: {_err}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach backend.")

    with _exp_col_clip:
        if st.session_state.get("export_json_text"):
            with st.expander("📋 Copy JSON to clipboard"):
                st.code(st.session_state["export_json_text"][:3000], language="json")
                st.caption("Use the copy icon above to copy the JSON to your clipboard.")

    if not _export_complete and run_id_for_export:
        st.caption("Export buttons are enabled once the evaluation run is complete.")
    elif not run_id_for_export:
        st.caption("Run an evaluation first to enable exports.")

    # ── 6. Compare Runs (Story 2.6) ───────────────────────────────────────
    st.divider()
    st.subheader("🔀 Compare Runs")
    st.caption("Select two completed runs to see side-by-side score and cost deltas.")

    try:
        hist_r = requests.get(f"{BACKEND_URL}/eval/history", timeout=5)
        if hist_r.status_code == 200:
            all_hist_runs = hist_r.json().get("runs", [])
            done_runs = [r for r in all_hist_runs if r["status"] == "complete"]
            if len(done_runs) < 2:
                st.info("Complete at least 2 eval runs to compare.")
            else:
                run_opts: dict = {}
                for r in done_runs:
                    lbl = r.get("run_label") or f"Run {r['id'][:8]}"
                    run_opts[f"{lbl} — {r['created_at']}"] = r["id"]

                cmp_col_a, cmp_col_b = st.columns(2)
                with cmp_col_a:
                    sel_a = st.selectbox("Run A", list(run_opts.keys()), key="cmp_run_a")
                with cmp_col_b:
                    default_b = min(1, len(run_opts) - 1)
                    sel_b = st.selectbox(
                        "Run B", list(run_opts.keys()), index=default_b, key="cmp_run_b"
                    )

                if st.button("Compare Runs", key="cmp_btn"):
                    aid = run_opts[sel_a]
                    bid = run_opts[sel_b]
                    cmp_r = requests.get(
                        f"{BACKEND_URL}/eval/compare",
                        params={"run_a": aid, "run_b": bid},
                        timeout=10,
                    )
                    if cmp_r.status_code == 200:
                        st.session_state.compare_data = cmp_r.json()
                    else:
                        try:
                            err_msg = cmp_r.json().get("detail", cmp_r.text)
                        except Exception:
                            err_msg = cmp_r.text
                        st.error(f"Compare failed: {err_msg}")

                if st.session_state.get("compare_data"):
                    _render_compare(st.session_state.compare_data)
        else:
            st.caption("Could not load run history for comparison.")
    except requests.exceptions.ConnectionError:
        st.caption("Backend offline — comparison unavailable.")
