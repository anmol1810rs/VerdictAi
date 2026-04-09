"""
VerdictAI — Streamlit frontend skeleton.
Session 1: layout, API key entry (Story 1.2), file upload (Story 1.1),
           rubric configurator (Story 1.3).
"""
import io
import os

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
DEV_MOCK = models_cfg["dev_mode"]

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


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR — API Keys (Story 1.2)
# Keys are stored only in session_state — never written to disk or DB
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚖️ VerdictAI")
    st.caption("Run structured LLM evaluations in minutes.")
    st.divider()

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

    st.divider()
    st.caption("💡 Google Gemini has a **free tier** — 1,000 requests/day, no credit card required. Best entry point to try VerdictAI for free.")


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
        # CSV template
        csv_template = "prompt,expected_output,engineer_name\n" \
                       "What is the capital of France?,Paris,Alice\n" \
                       "Summarise this article in one sentence.,,Bob\n"
        st.download_button(
            "Download CSV template",
            data=csv_template,
            file_name="verdictai_template.csv",
            mime="text/csv",
        )
        # JSONL template
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

                    # Show success + summary
                    st.success(f"✓ {meta.get('validation_summary', 'Dataset validated')}")

                    # Show warnings if any
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

    # Load preset weights or use defaults
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
        # Validate via backend
        try:
            val = requests.post(f"{BACKEND_URL}/rubric/validate", json=rubric, timeout=5)
            if val.status_code == 200:
                st.session_state.rubric = rubric
                st.success("Rubric valid ✓")
            else:
                st.error(f"Rubric rejected: {val.json()}")
        except requests.exceptions.ConnectionError:
            # Store locally if backend unreachable during dev
            st.session_state.rubric = rubric
    else:
        st.warning(f"Weights must sum to 100. Current: {total}. Adjust the sliders.")
        st.session_state.rubric = None


# ──────────────────────────────────────────────────────────────────────────
# TAB 3 — Model Selection (Story 1.5: modality filtering)
# ──────────────────────────────────────────────────────────────────────────

with tab_models:
    st.header("Select Models to Compare")
    st.caption("Compatible models shown based on your dataset's modality.")

    # Get detected modality from session state (from upload)
    detected_modality = st.session_state.get("detected_modality", "text")

    if st.session_state.upload_meta:
        st.info(f"📊 **Detected modality:** {detected_modality}")

        # Fetch compatible models from backend
        with st.spinner("Loading compatible models..."):
            try:
                compat_resp = requests.get(
                    f"{BACKEND_URL}/models/compatible",
                    params={"modality": detected_modality},
                    timeout=10,
                )
                if compat_resp.status_code == 200:
                    compat_data = compat_resp.json()
                    compatible_ids = [m["id"] for m in compat_data["compatible_models"]]
                    incompatible_list = compat_data.get("incompatible_models", [])
                    suggestions = compat_data.get("suggestions", {})

                    # Show compatible models as checkboxes
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

                    # Show incompatible with warnings + suggestions
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

    # Free tier note
    st.divider()
    st.caption(
        "💡 **Free tier available:** Gemini 2.5 Flash offers 1,000 requests/day at no cost. "
        "Best entry point for testing VerdictAI without spending on API credits."
    )


# ──────────────────────────────────────────────────────────────────────────
# TAB 4 — Run Evaluation
# ──────────────────────────────────────────────────────────────────────────

with tab_run:
    st.header("Run Evaluation")

    models = st.session_state.get("selected_models", [])
    prompts = st.session_state.uploaded_prompts
    rubric = st.session_state.rubric

    # Pre-run checklist
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

    # Story 1.4 — Engineer name (per-prompt, from upload) + Run label (optional, per-run)
    engineer_name = st.text_input(
        "Prompt engineer name (optional)",
        placeholder="e.g. Alice",
        help="This tags ALL prompts in this batch to a team member. (Optional)",
    )

    run_label = st.text_input(
        "Run label (optional)",
        placeholder="e.g. Sprint 4 QA or April batch 1",
        help="Add a descriptive label to identify this evaluation run later.",
        key="run_label_input",
    )

    all_ready = all(checks.values())

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

        with st.spinner("Running evaluation..."):
            try:
                resp = requests.post(f"{BACKEND_URL}/eval/run", json=payload, timeout=120)
                if resp.status_code == 200:
                    run_id = resp.json()["run_id"]
                    st.session_state.last_run_id = run_id
                    st.success(f"Evaluation complete! Run ID: `{run_id}`")
                    st.info("Switch to the **Results** tab to view scores and verdict.")
                else:
                    st.error(f"Eval failed: {resp.json()}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach backend. Is the FastAPI server running?")

    if not all_ready:
        st.caption("Complete the steps above to enable the run button.")


# ──────────────────────────────────────────────────────────────────────────
# TAB 5 — Results (placeholder — full implementation in Sessions 4-6)
# ──────────────────────────────────────────────────────────────────────────

with tab_results:
    st.header("Results")

    run_id = st.session_state.get("last_run_id")
    if not run_id:
        st.info("No evaluation run yet. Complete an eval in the **Run Evaluation** tab.")
    else:
        with st.spinner("Loading results..."):
            try:
                resp = requests.get(f"{BACKEND_URL}/eval/{run_id}/results", timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Run `{run_id}` — Status: **{data['status']}**")

                    if data.get("verdict"):
                        st.subheader("⚖️ Verdict")
                        st.markdown(f"**Winning model:** {data['verdict']['winning_model']}")
                        st.markdown(data["verdict"]["summary"])

                    st.subheader("📊 Scores (per result)")
                    st.caption("Full dimension scoring UI — Session 4")
                    for r in data["results"][:3]:
                        with st.expander(f"{r['model_name']} — prompt {r['prompt_index']}"):
                            st.json(r["dimension_scores"])
                else:
                    st.error(f"Could not load results: {resp.json()}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach backend.")
