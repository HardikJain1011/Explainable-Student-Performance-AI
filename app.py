"""Explainable Student Performance AI — portfolio-ready Streamlit app."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.inspection import permutation_importance

from src import explain as xai
from src.data import CLASS_NAMES, FEATURE_LABELS, FEATURES
from src.train import MODEL_DIR
from src.train import main as train_models
from src.recommendations import find_counterfactuals, personalized_suggestions
from src.report import build_report


st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Visual system
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background: #f6f8fb; }
    .block-container { max-width: 1220px; padding-top: 1.6rem; padding-bottom: 4rem; }
    .hero { padding: 2.2rem 2.4rem; border-radius: 26px; background: linear-gradient(135deg,#0f172a,#26364d); color:white; margin-bottom:1.2rem; box-shadow:0 16px 40px rgba(15,23,42,.14); }
    .hero h1 { font-size:2.55rem; margin:0 0 .45rem; letter-spacing:-.045em; color: #E8EEF7 !important; }
    .hero p { color:#dbe4ef; font-size:1.02rem; margin:0; max-width:850px; }
    .eyebrow { text-transform:uppercase; letter-spacing:.13em; font-size:.7rem; font-weight:800; color:#64748b; }
    .hero .eyebrow { color:#93c5fd; }
    .card { background:white; border:1px solid #e5e7eb; border-radius:18px; padding:1.15rem 1.25rem; box-shadow:0 6px 20px rgba(15,23,42,.045); }
    .dashboard-card { background:white; border:1px solid #e5e7eb; border-radius:20px; padding:1.2rem; min-height:145px; box-shadow:0 6px 20px rgba(15,23,42,.045); }
    .dashboard-value { font-size:1.85rem; font-weight:850; color:#111827; margin:.15rem 0 .1rem; }
    .dashboard-label { color:#64748b; font-size:.82rem; }
    .result-title { font-size:2.15rem; font-weight:850; margin:.2rem 0; color:#111827; }
    .muted { color:#64748b; }
    .pill { display:inline-block; padding:.28rem .62rem; border-radius:999px; background:#f1f5f9; color:#334155; font-size:.76rem; font-weight:700; margin:.12rem .2rem .12rem 0; }
    .section-title { font-size:1.45rem; font-weight:850; color:#111827; margin:.3rem 0 .2rem; }
    .mini-title { font-size:1.05rem; font-weight:800; color:#111827; }
    div[data-testid="stMetric"] { background:#fff; border:1px solid #e5e7eb; padding:1rem; border-radius:16px; }
    /* Keep Streamlit's native widget labels readable on the light app background. */
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stRadio"] label {
        color:#0f172a !important;
        font-weight:750 !important;
        opacity:1 !important;
    }
    div[data-testid="stWidgetLabel"] { margin-bottom:.2rem; }
    div[data-testid="stWidgetLabel"] p { font-size:.92rem !important; }
    div[data-testid="stTextInput"] input { color:#111827 !important; background:#ffffff !important; }
    div[data-testid="stTextInput"] input::placeholder { color:#94a3b8 !important; opacity:1 !important; }
    .stSlider [data-baseweb="slider"] { margin-top:.15rem; }
    .input-help { color:#475569 !important; font-size:.78rem; line-height:1.35; margin-top:-.3rem; margin-bottom:.85rem; }
    .input-help strong { color:#1e293b !important; }
    .input-summary { background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:.7rem .8rem; color:#475569; font-size:.78rem; line-height:1.4; margin-top:.4rem; }
    .stCaption, [data-testid="stCaptionContainer"] p { color:#64748b !important; opacity:1 !important; }
    /* Global readability: force readable text on the light app background. */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] span {
        color:#1e293b !important; opacity:1 !important;
    }
    h1, h2, h3, h4, h5, h6,
    [data-testid="stHeadingWithActionElements"] h1,
    [data-testid="stHeadingWithActionElements"] h2,
    [data-testid="stHeadingWithActionElements"] h3,
    [data-testid="stHeadingWithActionElements"] h4 {
        color:#0f172a !important; opacity:1 !important;
    }
    .section-title, .mini-title, .result-title, .eyebrow, .dashboard-label, .muted { opacity:1 !important; }
    .section-title, .mini-title { color:#0f172a !important; }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] { color:#0f172a !important; opacity:1 !important; }
    .stDataFrame, [data-testid="stDataFrame"] { color:#0f172a !important; }
    .js-plotly-plot .plotly .legendtext, .js-plotly-plot .plotly .xtick text, .js-plotly-plot .plotly .ytick text, .js-plotly-plot .plotly .gtitle { fill:#334155 !important; }
    .report-preview { background:#ffffff; border:1px solid #dbe3ec; border-radius:20px; padding:1.4rem; box-shadow:0 8px 28px rgba(15,23,42,.06); }
    .report-name { font-size:1.55rem; font-weight:850; color:#0f172a !important; margin-bottom:.15rem; }
    .report-kicker { color:#64748b !important; font-size:.82rem; margin-bottom:1rem; }
    .report-result { background:#f8fafc; border:1px solid #e2e8f0; border-radius:14px; padding:1rem; }
    .report-result-label { color:#64748b !important; font-size:.76rem; text-transform:uppercase; letter-spacing:.08em; font-weight:800; }
    .report-result-value { color:#0f172a !important; font-size:1.5rem; font-weight:850; }
    .report-section { color:#0f172a !important; font-size:1.05rem; font-weight:800; margin:1rem 0 .45rem; }
    .report-factor { padding:.55rem .7rem; border-radius:10px; background:#f8fafc; margin:.35rem 0; color:#334155 !important; }
    .report-note { color:#64748b !important; font-size:.78rem; line-height:1.45; }
    .stButton > button { border-radius:12px; font-weight:750; min-height:2.8rem; }
    .stDownloadButton > button { border-radius:12px; font-weight:750; }
    .stTabs [data-baseweb="tab-list"] { gap:.45rem; }
    .stTabs [data-baseweb="tab"] { border-radius:10px; padding:.6rem .9rem; }
    /* Expander: keep header light and its title readable in all states */
    [data-testid="stExpander"] details {
        background:#ffffff !important;
        border:1px solid #e2e8f0 !important;
        border-radius:14px !important;
    }
    [data-testid="stExpander"] details summary,
    [data-testid="stExpander"] details[open] summary,
    [data-testid="stExpander"] details summary:hover,
    [data-testid="stExpander"] details summary:focus,
    [data-testid="stExpander"] details summary:active {
        background:#f1f5f9 !important;
        color:#0f172a !important;
    }
    [data-testid="stExpander"] details summary p,
    [data-testid="stExpander"] details summary span,
    [data-testid="stExpander"] details summary [data-testid="stMarkdownContainer"] p {
        color:#0f172a !important;
        font-weight:750 !important;
        opacity:1 !important;
    }
    [data-testid="stExpander"] details summary svg {
        color:#0f172a !important;
        fill:#0f172a !important;
    }
    /* Hero is intentionally dark: keep its text bright even though the app
       uses dark text globally on light backgrounds. */
    .hero h1,
    .hero h1 span {
    color:#E8EEF7 !important;
    }
    .hero h2,
    .hero h2 span,
    .hero h3,
    .hero h3 span,
    .hero p,
    .hero p span,
    .hero .eyebrow {
        color:#E8EEF7 !important;
        opacity:1 !important;
    }
    .hero .eyebrow { color:#93c5fd !important; }
    .hero p { color:#dbe4ef !important; }
    .hero a { color:#93c5fd !important; }

    /* Streamlit/Plotly can inherit very low-contrast text from the host theme. */
    .js-plotly-plot .plotly text { fill:#334155 !important; }
    .js-plotly-plot .plotly .legendtext { fill:#334155 !important; }
    .js-plotly-plot .plotly .axis-title { fill:#475569 !important; }
    .js-plotly-plot .plotly .xtick text, .js-plotly-plot .plotly .ytick text { fill:#334155 !important; }
    .js-plotly-plot .plotly .gtitle { fill:#0f172a !important; }
    .js-plotly-plot .plotly .colorbar text { fill:#334155 !important; }

    /* Make the page navigation readable in both light and dark Streamlit themes. */
    div[data-testid="stRadio"] label p, div[data-testid="stRadio"] label span {
        color:#334155 !important; opacity:1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_plotly(fig, *, height=None, title=None):
    """Apply an explicit high-contrast Plotly theme so labels remain readable."""
    kwargs = {
        "plot_bgcolor": "#ffffff",
        "paper_bgcolor": "#ffffff",
        "font": {"family": "Inter, Arial, sans-serif", "size": 13, "color": "#334155"},
        # Hover tooltip styling
        "hoverlabel": {"bgcolor": "#ffffff","bordercolor": "#cbd5e1","font": {"family": "Inter, Arial, sans-serif","size": 13,"color": "#0f172a"}},
        "legend": {"font": {"size": 12, "color": "#334155"}},
        "xaxis": {
            "title": {"font": {"size": 13, "color": "#334155"}},
            "tickfont": {"size": 12, "color": "#334155"},
            "showgrid": True, "gridcolor": "#e2e8f0",
            "zerolinecolor": "#cbd5e1",
        },
        "yaxis": {
            "title": {"font": {"size": 13, "color": "#334155"}},
            "tickfont": {"size": 12, "color": "#334155"},
            "showgrid": True, "gridcolor": "#e2e8f0",
            "zerolinecolor": "#cbd5e1",
        },
    }
    if height is not None:
        kwargs["height"] = height
    if title is not None:
        kwargs["title"] = {"text": title, "font": {"size": 18, "color": "#0f172a"}}
    fig.update_layout(**kwargs)
    return fig


@st.cache_resource(show_spinner="Preparing the prediction engine...")
def load_artifacts():
    if not (MODEL_DIR / "meta.json").exists():
        train_models()
    models = joblib.load(MODEL_DIR / "all_models.joblib")
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    metrics = pd.read_csv(MODEL_DIR / "metrics.csv", index_col=0)
    X_train = pd.read_csv(MODEL_DIR / "X_train.csv")
    X_test = pd.read_csv(MODEL_DIR / "X_test.csv")
    y_test = pd.read_csv(MODEL_DIR / "y_test.csv").iloc[:, 0]
    return models, meta, metrics, X_train, X_test, y_test


@st.cache_resource
def get_explainers(model_name):
    models, _, _, X_train, _, _ = load_artifacts()
    model = models[model_name]
    bg = X_train.sample(min(200, len(X_train)), random_state=0)
    return xai.build_shap_explainer(model, bg), xai.build_lime_explainer(X_train, CLASS_NAMES)


@st.cache_data(show_spinner="Computing global explanations...")
def global_shap(model_name):
    models, _, _, _, X_test, _ = load_artifacts()
    bundle, _ = get_explainers(model_name)
    sample = X_test.sample(min(300, len(X_test)), random_state=0)
    return sample, xai.shap_values(bundle, sample)


def contribution_chart(table, title):
    t = table.head(7).iloc[::-1]
    labels = [f"{FEATURE_LABELS[f]} = {v:g}" for f, v in zip(t.feature, t.value)]
    fig = go.Figure(go.Bar(
        x=t.shap, y=labels, orientation="h",
        marker_color=["#16a34a" if s > 0 else "#dc2626" for s in t.shap],
        hovertemplate="%{y}<br>Impact: %{x:.3f}<extra></extra>",
    ))
    style_plotly(fig, height=390, title=title)
    fig.update_layout(margin=dict(l=8,r=16,t=58,b=12), xaxis_title="SHAP impact")
    return fig


def format_change(feature, old, new):
    if feature == "extracurriculars":
        return f"{FEATURE_LABELS[feature]}: {old:.0f} → {new:.0f}"
    if feature in {"attendance", "assignments_completed"}:
        return f"{FEATURE_LABELS[feature]}: {old:.0f}% → {new:.0f}%"
    return f"{FEATURE_LABELS[feature]}: {old:.1f} → {new:.1f}"


def render_dashboard(inputs, pred, confidence, proba, table, suggestions):
    st.markdown("### Student dashboard")
    st.caption("A concise view of the model assessment and the factors you can explore next.")

    positive = table[table.shap > 0].head(1)
    negative = table[table.shap < 0].head(1)
    strongest_positive = FEATURE_LABELS[positive.iloc[0].feature] if not positive.empty else "None identified"
    strongest_negative = FEATURE_LABELS[negative.iloc[0].feature] if not negative.empty else "None identified"
    priority = suggestions[0][1] if suggestions else "Maintain current strengths"

    cols = st.columns(4)
    cards = [
        ("Prediction", pred, f"{confidence:.0%} model probability"),
        ("Strongest positive", strongest_positive, "Largest positive SHAP contributor"),
        ("Priority area", priority, "Based on current model explanation"),
        ("Largest negative", strongest_negative, "Largest negative SHAP contributor"),
    ]
    for col, (label, value, sub) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="dashboard-card"><div class="eyebrow">{label}</div>'
                f'<div class="dashboard-value">{value}</div><div class="dashboard-label">{sub}</div></div>',
                unsafe_allow_html=True,
            )

    left, right = st.columns([1.15, .85], gap="large")
    with left:
        st.markdown("#### Performance outlook")
        probability_df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": proba})
        fig = px.bar(probability_df, x="Class", y="Probability", text_auto=".0%", range_y=[0,1])
        fig.update_traces(marker_color=["#94a3b8", "#64748b", "#16a34a"])
        fig.update_layout(height=310, yaxis_tickformat=".0%", plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=0,r=0,t=20,b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown("#### Student profile")
        profile = pd.DataFrame({
            "Factor": [FEATURE_LABELS[f] for f in FEATURES],
            "Value": [inputs[f] for f in FEATURES],
        })
        st.dataframe(profile, hide_index=True, use_container_width=True, height=300)

    st.markdown("#### What the model is seeing")
    a, b = st.columns(2)
    with a:
        st.markdown("**🟢 Helping the prediction**")
        for _, r in table[table.shap > 0].head(3).iterrows():
            st.write(f"{FEATURE_LABELS[r.feature]} · {r.value:g}")
    with b:
        st.markdown("**🔴 Pulling the prediction down**")
        for _, r in table[table.shap < 0].head(3).iterrows():
            st.write(f"{FEATURE_LABELS[r.feature]} · {r.value:g}")


def render_report_preview(name, inputs, pred, confidence, proba, table, suggestions, candidates):
    """Render the core content that also appears in the downloadable PDF."""
    st.markdown("### Student report")
    st.markdown(
        f'''<div class="report-preview">
        <div class="report-name">🎓 {name}'s Performance Report</div>
        <div class="report-kicker">Explainable AI assessment · XGBoost + SHAP + LIME</div>
        <div class="report-result">
          <div class="report-result-label">Predicted performance</div>
          <div class="report-result-value">{pred} · {confidence:.0%} model probability</div>
        </div>
        </div>''', unsafe_allow_html=True,
    )
    st.markdown("#### Performance outlook")
    prob_cols = st.columns(3)
    for i, cls in enumerate(CLASS_NAMES):
        with prob_cols[i]:
            st.markdown(
                f'''<div class="report-result"><div class="report-result-label">{cls}</div>
                <div class="report-result-value">{float(proba[i]):.0%}</div></div>''',
                unsafe_allow_html=True,
            )

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="report-section">Student profile</div>', unsafe_allow_html=True)
        profile_rows = "".join(
            f'<div class="report-factor"><b>{FEATURE_LABELS[f]}</b>: {v:g}</div>'
            for f, v in inputs.items()
        )
        st.markdown(profile_rows, unsafe_allow_html=True)
    with right:
        st.markdown('<div class="report-section">What influenced the prediction</div>', unsafe_allow_html=True)
        positive = table[table.shap > 0].head(3)
        negative = table[table.shap < 0].head(3)
        if not positive.empty:
            st.markdown("**🟢 Helping**")
            for _, r in positive.iterrows():
                st.markdown(f'<div class="report-factor"><b>{FEATURE_LABELS[r.feature]}</b> · {r.value:g} · impact {r.shap:+.3f}</div>', unsafe_allow_html=True)
        if not negative.empty:
            st.markdown("**🔴 Pulling it down**")
            for _, r in negative.iterrows():
                st.markdown(f'<div class="report-factor"><b>{FEATURE_LABELS[r.feature]}</b> · {r.value:g} · impact {r.shap:+.3f}</div>', unsafe_allow_html=True)

    st.markdown('<div class="report-section">Personalized improvement plan</div>', unsafe_allow_html=True)
    if suggestions:
        for _, title_text, target, description in suggestions[:3]:
            st.markdown(f'<div class="report-factor"><b>{title_text}</b> · {target}<br><span class="report-note">{description}</span></div>', unsafe_allow_html=True)
    else:
        st.info("No strong actionable negative contributor was identified for this profile.")

    if candidates:
        st.markdown('<div class="report-section">What-if scenarios</div>', unsafe_allow_html=True)
        for i, item in enumerate(candidates[:3], 1):
            changes = [format_change(f, float(inputs[f]), float(item["candidate"][f])) for f in item["changes"]]
            change_text = " · ".join(changes)
            st.markdown(f'<div class="report-factor"><b>Option {i}</b> · simulated High probability: {item["confidence"]:.0%}<br><span class="report-note">{change_text}</span></div>', unsafe_allow_html=True)

    st.caption("Model-based assessment only. This report is not a causal claim, diagnosis, or substitute for academic advising.")


def render_improvement_plan(model, x_row, table, pred_idx):
    st.markdown("### Improvement plan")
    st.caption("Model-grounded suggestions are what-if scenarios, not guarantees or causal advice.")
    suggestions = personalized_suggestions(model, x_row, table, pred_idx)

    if pred_idx == 2:
        st.success("The model currently places this profile in the High group. Focus on maintaining the factors that are helping the prediction.")
    elif suggestions:
        cols = st.columns(min(2, len(suggestions)))
        for i, (_, title, target, description) in enumerate(suggestions):
            with cols[i % len(cols)]:
                st.markdown(f'<div class="card"><div class="mini-title">{title}</div><b>{target}</b><br><span class="muted">{description}</span></div>', unsafe_allow_html=True)
    else:
        st.info("No strong directly actionable negative contributor was identified for this profile.")

    candidates = []
    if pred_idx < 2:
        st.markdown("#### What-if simulator")
        with st.spinner("Searching for small model-based changes..."):
            candidates = find_counterfactuals(model, x_row, desired_class=2, max_changes=3)
        if candidates:
            for i, item in enumerate(candidates[:3], 1):
                candidate = item["candidate"]
                changes = [format_change(f, float(x_row[f]), float(candidate[f])) for f in item["changes"]]
                st.markdown(f"**Option {i}** · simulated High probability: **{item['confidence']:.0%}**")
                st.write(" · ".join(changes))
                st.progress(float(item["confidence"]))
        else:
            st.info("No nearby combination of up to three actionable changes moved this profile to High in the simulator.")
    return suggestions, candidates


def render_xai(model, model_name, x_df, x_row, pred_idx, pred):
    shap_bundle, lime_exp = get_explainers(model_name)
    sv = xai.shap_values(shap_bundle, x_df)[0]
    table = xai.contribution_table(sv[:, pred_idx], x_row)

    st.markdown("### Why did the model predict this?")
    st.caption("SHAP estimates how each input moved the model's score for the predicted class relative to its baseline.")
    st.plotly_chart(contribution_chart(table, f"Drivers of the {pred} prediction"), use_container_width=True, config={"displayModeBar": False})

    a, b = st.columns(2)
    with a:
        st.markdown("#### 🟢 Helping")
        for _, r in table[table.shap > 0].head(4).iterrows():
            st.write(f"**{FEATURE_LABELS[r.feature]}** · {r.value:g} · impact `{r.shap:+.3f}`")
    with b:
        st.markdown("#### 🔴 Pulling it down")
        for _, r in table[table.shap < 0].head(4).iterrows():
            st.write(f"**{FEATURE_LABELS[r.feature]}** · {r.value:g} · impact `{r.shap:+.3f}`")

    with st.expander("🔎 SHAP vs LIME local explanation"):
        st.caption("LIME provides a second local explanation method. Agreement is a useful sanity check, not proof of causal truth.")
        with st.spinner("Generating LIME explanation..."):
            lime_list = xai.lime_explain(lime_exp, model, x_row, pred_idx)
        ldf = pd.DataFrame(lime_list, columns=["rule", "weight"]).iloc[::-1]
        fig = go.Figure(go.Bar(x=ldf.weight, y=ldf.rule, orientation="h",
                               marker_color=["#16a34a" if w > 0 else "#dc2626" for w in ldf.weight]))
        style_plotly(fig, height=360)
        fig.update_layout(margin=dict(l=0,r=0,t=10,b=10), xaxis_title="LIME weight")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    return table


# ---------------------------------------------------------------------------
# Load model assets
# ---------------------------------------------------------------------------
models, meta, metrics, X_train, X_test, y_test = load_artifacts()
model_name = meta["best_model"]
model = models[model_name]

st.markdown(
    '<div class="hero"><div class="eyebrow">Explainable AI · Student Success</div>'
    '<h1>🎓 Student Performance AI</h1>'
    '<p>Predict performance, understand the model, and explore practical what-if improvements through an interactive XAI dashboard.</p></div>',
    unsafe_allow_html=True,
)

page = st.radio("Navigation", ["🔮 Prediction", "📈 Model Insights", "🧠 XAI Lab", "ℹ️ About"], horizontal=True, label_visibility="collapsed")


# ---------------------------------------------------------------------------
# Prediction / student dashboard
# ---------------------------------------------------------------------------
if page == "🔮 Prediction":
    st.markdown('<div class="section-title">Build a student profile</div>', unsafe_allow_html=True)
    st.caption("Enter the student's current profile, then generate a model assessment and an explainable student dashboard.")

    with st.form("student_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            student_name = st.text_input("Student name (optional)", placeholder="e.g. Alex")
            st.caption("Your name is only used to personalize the dashboard and report.")

            attendance = st.slider("Attendance (%)", 30, 100, 80)
            st.markdown("<div class='input-help'><strong>What this means:</strong> Percentage of classes attended.</div>", unsafe_allow_html=True)

            prev_gpa = st.slider("Previous semester GPA (out of 10)", 4.0, 10.0, 7.5, 0.1)
            st.markdown("<div class='input-help'><strong>What this means:</strong> Your GPA from the previous semester.</div>", unsafe_allow_html=True)

        with c2:
            study_hours = st.slider("Study hours per day", 0.0, 10.0, 3.5, 0.5)
            st.markdown("<div class='input-help'><strong>What this means:</strong> Average number of hours spent studying each day.</div>", unsafe_allow_html=True)

            assignments = st.slider("Assignments completed (%)", 20, 100, 85)
            st.markdown("<div class='input-help'><strong>What this means:</strong> Approximate percentage of assigned work completed.</div>", unsafe_allow_html=True)

            sleep_hours = st.slider("Sleep hours per night", 3.0, 10.0, 7.0, 0.5)
            st.markdown("<div class='input-help'><strong>What this means:</strong> Average hours of sleep you get each night.</div>", unsafe_allow_html=True)

        with c3:
            extracurriculars = st.slider("Extracurricular activities (count)", 0, 6, 2)
            st.markdown("<div class='input-help'><strong>What this means:</strong> Number of regular extracurricular activities you participate in.</div>", unsafe_allow_html=True)

            prev_exam_score = st.slider("Previous exam score (%)", 20, 100, 70)
            st.markdown("<div class='input-help'><strong>What this means:</strong> Your score on a previous exam, as a percentage.</div>", unsafe_allow_html=True)

            st.markdown("<div class='input-summary'><strong>7 factors</strong> used by the model · XGBoost prediction · SHAP + LIME explanations</div>", unsafe_allow_html=True)
            st.caption("For learning and decision support only — not for high-stakes academic decisions.")
        submitted = st.form_submit_button("✨ Generate student dashboard", type="primary", use_container_width=True)

    if submitted or "assessment" not in st.session_state:
        if submitted:
            st.session_state.assessment = {
                "name": student_name.strip() or "Student",
                "inputs": {
                    "attendance": attendance, "prev_gpa": prev_gpa, "study_hours": study_hours,
                    "assignments_completed": assignments, "sleep_hours": sleep_hours,
                    "extracurriculars": extracurriculars, "prev_exam_score": prev_exam_score,
                },
            }
        else:
            st.session_state.assessment = {
                "name": "Student",
                "inputs": {
                    "attendance": 80, "prev_gpa": 7.5, "study_hours": 3.5,
                    "assignments_completed": 85, "sleep_hours": 7.0,
                    "extracurriculars": 2, "prev_exam_score": 70,
                },
            }

        assessment = st.session_state.assessment
        x_row = pd.Series(assessment["inputs"], index=FEATURES, dtype=float)
        x_df = x_row.to_frame().T
        proba = model.predict_proba(x_df)[0]
        pred_idx = int(np.argmax(proba))
        pred = CLASS_NAMES[pred_idx]
        confidence = float(proba[pred_idx])
        shap_bundle, _ = get_explainers(model_name)
        sv = xai.shap_values(shap_bundle, x_df)[0]
        table = xai.contribution_table(sv[:, pred_idx], x_row)
        suggestions = personalized_suggestions(model, x_row, table, pred_idx)

        st.divider()
        st.markdown(f"### {assessment['name']}'s assessment")
        render_dashboard(assessment["inputs"], pred, confidence, proba, table, suggestions)

        candidates = find_counterfactuals(model, x_row, desired_class=2, max_changes=3) if pred_idx < 2 else []
        positive = list(table[table.shap > 0].head(4).itertuples(index=False))
        negative = list(table[table.shap < 0].head(4).itertuples(index=False))
        prob_map = {CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))}

        # Show the report immediately after the dashboard is generated.
        render_report_preview(assessment["name"], assessment["inputs"], pred, confidence, proba, table, suggestions, candidates)
        pdf = build_report(assessment["name"], assessment["inputs"], pred, confidence, prob_map, positive, negative, suggestions, candidates)
        st.download_button(
            "⬇️ Download this student report as PDF", data=pdf,
            file_name=f"{assessment['name'].replace(' ','_')}_performance_report.pdf",
            mime="application/pdf", type="primary", use_container_width=True,
        )

        tabs = st.tabs(["🧠 Explain the prediction", "🎯 Improvement plan"])
        with tabs[0]:
            render_xai(model, model_name, x_df, x_row, pred_idx, pred)

        with tabs[1]:
            render_improvement_plan(model, x_row, table, pred_idx)


# ---------------------------------------------------------------------------
# Model insights
# ---------------------------------------------------------------------------
elif page == "📈 Model Insights":
    st.markdown('<div class="section-title">Model performance</div>', unsafe_allow_html=True)
    st.caption("The production model is selected automatically by macro-F1 on the held-out test set.")
    best = metrics["F1"].idxmax()
    cols = st.columns(4)
    cols[0].metric("Selected model", best)
    cols[1].metric("Test F1", f"{metrics.loc[best, 'F1']:.3f}")
    cols[2].metric("Accuracy", f"{metrics.loc[best, 'Accuracy']:.3f}")
    cols[3].metric("ROC-AUC", f"{metrics.loc[best, 'ROC-AUC']:.3f}")
    st.markdown("#### Model comparison")
    st.dataframe(metrics.style.format("{:.3f}").highlight_max(axis=0), use_container_width=True)
    long = metrics.drop(columns=["CV F1 (5-fold)"]).reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group", range_y=[max(0, long.Score.min()-.05),1])
    style_plotly(fig, height=430)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.warning("The included dataset is synthetic. Strong benchmark scores validate the engineering pipeline, not real-world predictive validity.")


# ---------------------------------------------------------------------------
# XAI lab
# ---------------------------------------------------------------------------
elif page == "🧠 XAI Lab":
    st.markdown('<div class="section-title">Explainability laboratory</div>', unsafe_allow_html=True)
    st.caption("Technical views for demonstrating SHAP, permutation importance and feature dependence.")
    chosen = st.selectbox("Model to inspect", list(models), index=list(models).index(model_name))
    target_class = st.radio("Class", CLASS_NAMES, index=2, horizontal=True)
    sample, gsv = global_shap(chosen)
    ci = CLASS_NAMES.index(target_class)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Global SHAP summary")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7.2, 4.2), facecolor="white")
        ax = plt.gca()
        ax.set_facecolor("white")
        shap.summary_plot(gsv[:, :, ci], sample, show=False, plot_size=None)
        ax = plt.gca()
        ax.tick_params(axis="both", colors="#334155", labelsize=9)
        ax.xaxis.label.set_color("#334155")
        ax.yaxis.label.set_color("#334155")
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
        plt.tight_layout()
        st.pyplot(plt.gcf(), clear_figure=True)
    with right:
        st.markdown("#### SHAP vs permutation importance")
        mean_abs = np.abs(gsv[:, :, ci]).mean(axis=0)
        yt = y_test.loc[sample.index]
        perm = permutation_importance(models[chosen], sample, yt, scoring="f1_macro", n_repeats=10, random_state=0).importances_mean
        cmp = pd.DataFrame({"Feature":[FEATURE_LABELS[f] for f in FEATURES], "Mean |SHAP|":mean_abs/max(mean_abs.sum(),1e-9), "Permutation importance":np.clip(perm,0,None)/max(np.clip(perm,0,None).sum(),1e-9)}).sort_values("Mean |SHAP|")
        fig = px.bar(cmp.melt(id_vars="Feature"), x="value", y="Feature", color="variable", barmode="group", labels={"value":"Normalized importance","variable":""})
        style_plotly(fig, height=440)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("#### Feature dependence")
    feat = st.selectbox("Choose a feature", FEATURES, format_func=lambda f: FEATURE_LABELS[f])
    fi = FEATURES.index(feat)
    fig = px.scatter(x=sample[feat], y=gsv[:, fi, ci], labels={"x":FEATURE_LABELS[feat],"y":f"SHAP value for '{target_class}'"})
    fig.add_hline(y=0, line_dash="dash")
    style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# About / portfolio page
# ---------------------------------------------------------------------------
else:
    st.markdown('<div class="section-title">About the project</div>', unsafe_allow_html=True)
    st.write("An end-to-end Explainable AI application that predicts whether a student is likely to fall into Low, Medium, or High performance groups, explains individual predictions with SHAP and LIME, and explores model-based improvement scenarios.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### ML engineering")
        st.markdown("- Logistic Regression\n- Random Forest\n- XGBoost\n- Cross-validation\n- Held-out evaluation\n- Model artifact pipeline")
    with c2:
        st.markdown("#### Explainable AI")
        st.markdown("- Local SHAP attribution\n- LIME cross-check\n- Global SHAP\n- Permutation importance\n- Feature dependence\n- Counterfactual what-if analysis")
    with c3:
        st.markdown("#### Product layer")
        st.markdown("- Student dashboard\n- Probability visualization\n- Personalized improvement plan\n- Downloadable PDF report\n- Streamlit deployment\n- Portfolio-ready documentation")
    st.divider()
    st.markdown("#### Architecture")
    st.code("Student inputs → preprocessing/model → prediction probabilities → SHAP/LIME → actionable recommendations → dashboard + PDF report", language="text")
    st.markdown("#### Responsible AI & privacy")
    st.write(
        "This is an educational and portfolio-oriented prototype. The included dataset "
        "is synthetic and does not represent real students. Predictions are probabilistic "
        "and should not be treated as official academic assessments or used as the sole "
        "basis for consequential educational decisions."
    )
    st.write(
        "SHAP and LIME explain model behavior; they do not establish causal relationships. "
        "Counterfactual scenarios are model-based what-if simulations, not guaranteed "
        "interventions or academic advice."
    )
    st.write(
        "The public demo should not be used with personally identifiable or sensitive "
        "student records. If real student data is introduced in the future, appropriate "
        "consent, privacy, security, retention, fairness, validation and institutional "
        "requirements must be addressed."
    )
    st.markdown("#### Portfolio positioning")
    st.success("This project demonstrates applied machine learning, model evaluation, explainable AI, interactive product development and responsible AI communication in one end-to-end system.")

st.caption("Built with Python · scikit-learn · XGBoost · SHAP · LIME · Plotly · Streamlit")
