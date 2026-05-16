import streamlit as st
import requests
import json
import time
import re
import os
import subprocess
import sys

# ── Auto-Start Backend (for Streamlit Cloud Deployment) ──
if "backend_started" not in st.session_state:
    try:
        # Check if backend is already reachable
        requests.get("http://localhost:8000/", timeout=1)
    except:
        # If not, launch the FastAPI server in a background process
        # We use sys.executable to ensure we use the same Python environment
        subprocess.Popen([sys.executable, "-m", "backend.main"])
        time.sleep(5)  # Wait for engines to initialize (embeddings + ML models)
    st.session_state.backend_started = True

# ── Page Config ──
st.set_page_config(
    page_title="ResearchLens AI | Intelligent Research Mentor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS ──
css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Backend URL ──
API_URL = "http://localhost:8000"

# ── Session State ──
if "paper_id" not in st.session_state:
    st.session_state.paper_id = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "upload_info" not in st.session_state:
    st.session_state.upload_info = None


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚀 ResearchLens AI")
    st.markdown("*Intelligent Research Mentor*")
    st.markdown("---")

    # ── PDF Upload ──
    st.markdown("### 📄 Upload Research Paper")
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Supports IEEE, arXiv, Springer, ACM papers",
    )

    if uploaded_file and st.button("📤 Process Paper", use_container_width=True):
        with st.spinner("🔍 Parsing & indexing paper..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_URL}/upload", files=files, timeout=120)
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.paper_id = result["paper_id"]
                    st.session_state.upload_info = result
                    st.session_state.analysis = None
                    st.session_state.chat_history = []
                    st.success(f"✅ Processed! {result['chunks_indexed']} chunks indexed.")
                else:
                    st.error(f"❌ Upload failed: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Run: `python -m backend.main`")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # ── Active Paper Indicator ──
    if st.session_state.upload_info:
        info = st.session_state.upload_info
        title = info['metadata'].get('title')
        if not title:
            title = info['metadata'].get('filename', 'Unknown Paper')
        if title.lower().endswith('.pdf'):
            title = title[:-4]
        
        st.markdown("---")
        st.markdown("### 📄 Active Paper")
        st.markdown(
            f"""<div style="background:rgba(139,92,246,0.1); border:1px solid rgba(139,92,246,0.25); border-radius:8px; padding:12px; font-size:0.85rem; color:#F1F5F9; line-height:1.4;">
                {title}
            </div>""",
            unsafe_allow_html=True
        )

    # ── Run Analysis ──
    if st.session_state.paper_id:
        st.markdown("---")
        if st.button("🔥 Run Full AI Analysis", use_container_width=True):
            with st.spinner("🤖 Running AI + ML analysis... This may take 1-2 minutes."):
                try:
                    response = requests.get(
                        f"{API_URL}/analyze/{st.session_state.paper_id}",
                        timeout=300,
                    )
                    if response.status_code == 200:
                        st.session_state.analysis = response.json()
                        st.success("✅ Analysis complete!")
                    else:
                        st.error(f"❌ Analysis failed: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#64748B;font-size:0.75rem;'>"
        "Built with LangChain + Qdrant + Llama 3.1<br/>5-Model ML Ensemble</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════
if not st.session_state.paper_id:
    # ── Landing Page ──
    st.markdown(
        """<div style="text-align:center; padding:10px 20px 20px;">
            <div class="lp-hero-title">🚀 ResearchLens AI</div>
            <div style="width: 150px; height: 2px; background: linear-gradient(90deg, transparent, rgba(139,92,246,0.8), transparent); margin: 10px auto 1.5rem; border-radius: 2px;"></div>
            <div style="font-size:1.15rem; color:#94A3B8; max-width:550px; margin:0 auto 2.5rem; line-height:1.6;">
                Your AI-powered research mentor — upload any paper and get deep analysis,
                ML predictions, interactive quizzes, and a smart chatbot.
            </div>
            <div class="lp-feat-grid">
                <div class="lp-feat-card">
                    <div class="lp-feat-icon" style="background:rgba(139,92,246,0.15);color:#C4B5FD;">🧠</div>
                    <div class="lp-feat-title">AI Explainer</div>
                    <div class="lp-feat-desc">Deep paper analysis</div>
                </div>
                <div class="lp-feat-card">
                    <div class="lp-feat-icon" style="background:rgba(16,185,129,0.15);color:#6EE7B7;">📈</div>
                    <div class="lp-feat-title">5-Model Ensemble</div>
                    <div class="lp-feat-desc">ML predictions</div>
                </div>
                <div class="lp-feat-card">
                    <div class="lp-feat-icon" style="background:rgba(14,165,233,0.15);color:#7DD3FC;">💬</div>
                    <div class="lp-feat-title">RAG Chatbot</div>
                    <div class="lp-feat-desc">Ask anything</div>
                </div>
                <div class="lp-feat-card">
                    <div class="lp-feat-icon" style="background:rgba(245,158,11,0.15);color:#FCD34D;">➗</div>
                    <div class="lp-feat-title">Math Explainer</div>
                    <div class="lp-feat-desc">Equation breakdown</div>
                </div>
                <div class="lp-feat-card">
                    <div class="lp-feat-icon" style="background:rgba(239,68,68,0.15);color:#FCA5A5;">📝</div>
                    <div class="lp-feat-title">Interactive Quiz</div>
                    <div class="lp-feat-desc">Test understanding</div>
                </div>
                <div class="lp-feat-card">
                    <div class="lp-feat-icon" style="background:rgba(168,85,247,0.15);color:#D8B4FE;">🏗️</div>
                    <div class="lp-feat-title">Architecture Viz</div>
                    <div class="lp-feat-desc">Auto diagrams</div>
                </div>
            </div>
            <div style="display:flex;justify-content:center;gap:10px;flex-wrap:wrap;margin:1.5rem auto 1rem;">
                <span style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.25);border-radius:20px;padding:5px 14px;font-size:0.75rem;color:#A78BFA;font-weight:500;">🦙 Llama 3.1</span>
                <span style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.25);border-radius:20px;padding:5px 14px;font-size:0.75rem;color:#A78BFA;font-weight:500;">🔗 LangChain</span>
                <span style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.25);border-radius:20px;padding:5px 14px;font-size:0.75rem;color:#A78BFA;font-weight:500;">📦 Qdrant</span>
                <span style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.25);border-radius:20px;padding:5px 14px;font-size:0.75rem;color:#A78BFA;font-weight:500;">⚡ FastAPI</span>
                <span style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.25);border-radius:20px;padding:5px 14px;font-size:0.75rem;color:#A78BFA;font-weight:500;">🤖 5 ML Models</span>
            </div>
            <div style="color:#64748B;font-size:0.9rem;margin-top:0.5rem;">← Upload a research paper PDF in the sidebar to get started</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.stop()


if not st.session_state.analysis:
    info = st.session_state.upload_info or {}
    meta = info.get("metadata", {})

    st.markdown(
        f"""
        <div style="text-align:center; padding:40px 20px 20px;">
            <div style="font-size:3.5rem; margin-bottom:0.5rem;">🎉</div>
            <h2 style="margin-bottom:0.3rem;">Paper Processed Successfully!</h2>
            <p style="color:#94A3B8; font-size:1.1rem; margin-bottom:2rem;">
                Your paper is now securely indexed in the vector database and ready for deep AI analysis.
            </p>
        </div>

        <div style="max-width:700px; margin:0 auto 2rem;">
            <div style="background:rgba(22,33,62,0.4); border:1px solid rgba(255,255,255,0.08); border-radius:12px; overflow:hidden;">
                <table style="width:100%; border-collapse:collapse; text-align:left; font-size:0.95rem; color:#cbd5e1;">
                    <thead>
                        <tr style="background:rgba(139,92,246,0.15); border-bottom:1px solid rgba(139,92,246,0.3);">
                            <th colspan="2" style="padding:12px 20px; font-weight:600; color:#F1F5F9;">📄 Paper Details</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);"><td style="padding:12px 20px; width:40%;">📄 Title</td><td style="padding:12px 20px; color:#A78BFA; font-weight:600;">{(meta.get('title') or meta.get('filename', 'Unknown Paper')).replace('.pdf', '').replace('.PDF', '')}</td></tr>
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);"><td style="padding:12px 20px;">📑 Pages</td><td style="padding:12px 20px; color:#38BDF8; font-weight:600;">{meta.get('page_count', 'N/A')}</td></tr>
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);"><td style="padding:12px 20px;">📂 Sections Detected</td><td style="padding:12px 20px; color:#34D399; font-weight:600;">{len(info.get('sections', []))}</td></tr>
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);"><td style="padding:12px 20px;">📐 Equations Found</td><td style="padding:12px 20px; color:#FBBF24; font-weight:600;">{info.get('equation_count', 0)}</td></tr>
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);"><td style="padding:12px 20px;">📚 References</td><td style="padding:12px 20px; color:#F87171; font-weight:600;">{info.get('reference_count', 0)}</td></tr>
                        <tr><td style="padding:12px 20px;">🧩 Chunks Indexed in Qdrant</td><td style="padding:12px 20px; color:#C084FC; font-weight:600;">{info.get('chunks_indexed', 0)}</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div style="text-align:center; margin:1rem auto; max-width:650px;">
            <div style="background:linear-gradient(145deg, rgba(22,33,62,0.8), rgba(15,23,42,0.9)); border:1px solid rgba(139,92,246,0.3); border-radius:16px; padding:32px; box-shadow:0 10px 30px rgba(0,0,0,0.2);">
                <div style="font-size:1.3rem; font-weight:700; color:#F1F5F9; margin-bottom:20px;">
                    🚀 Ready to unlock the insights?
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; text-align:left; font-size:0.95rem; color:#cbd5e1; margin-bottom:24px;">
                    <div style="background:rgba(139,92,246,0.1); padding:10px 14px; border-radius:8px;">🧠 <strong>Deep Summary</strong> & Concepts</div>
                    <div style="background:rgba(16,185,129,0.1); padding:10px 14px; border-radius:8px;">🤖 <strong>5-Model</strong> ML Predictions</div>
                    <div style="background:rgba(14,165,233,0.1); padding:10px 14px; border-radius:8px;">💬 <strong>Smart Chatbot</strong> (RAG)</div>
                    <div style="background:rgba(245,158,11,0.1); padding:10px 14px; border-radius:8px;">➗ <strong>Math</strong> Equation Breakdown</div>
                    <div style="background:rgba(239,68,68,0.1); padding:10px 14px; border-radius:8px;">📝 <strong>Interactive Quiz</strong></div>
                    <div style="background:rgba(168,85,247,0.1); padding:10px 14px; border-radius:8px;">🏗️ <strong>Architecture</strong> Viz</div>
                </div>
                <div style="color:#94A3B8; font-size:0.95rem; margin-top:10px; background:rgba(0,0,0,0.2); padding:12px; border-radius:8px;">
                    Click the <strong style="color:#A78BFA;">🔥 Run Full AI Analysis</strong> button in the sidebar to begin!
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════
#  ANALYSIS DASHBOARD
# ══════════════════════════════════════════════════════════════════
analysis = st.session_state.analysis

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard",
    "➗ Math & Concepts",
    "🔬 Innovation",
    "🛠️ Implementation",
    "📝 Quiz",
    "💬 Chat",
])

# ── TAB 1: Dashboard ──
with tab1:
    st.markdown("## 📊 Analysis Dashboard")

    # ML Score Cards
    col1, col2, col3, col4 = st.columns(4)

    ml_repro = analysis.get("ml_reproducibility", {})
    ml_diff = analysis.get("ml_difficulty", {})

    repro_score = ml_repro.get("ensemble_score", "N/A")
    diff_level = ml_diff.get("ensemble_level", "N/A")

    with col1:
        score_class = "score-high" if isinstance(repro_score, (int, float)) and repro_score >= 7 else "score-mid" if isinstance(repro_score, (int, float)) and repro_score >= 4 else "score-low"
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value {score_class}">{repro_score}/10</div>
                <div class="metric-label">Reproducibility Score</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col2:
        diff_emoji = "🟢" if diff_level == "Beginner" else "🟡" if diff_level == "Intermediate" else "🔴"
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value" style="font-size:1.8rem;">{diff_emoji} {diff_level}</div>
                <div class="metric-label">Difficulty Level</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col3:
        info = st.session_state.upload_info or {}
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value" style="font-size:1.8rem;">📄 {info.get('metadata', {}).get('page_count', '?')}</div>
                <div class="metric-label">Pages</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value" style="font-size:1.8rem;">📐 {info.get('equation_count', '?')}</div>
                <div class="metric-label">Equations Detected</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Summary
    st.markdown("### 📖 Paper Summary")
    summary = analysis.get("summary", {})
    if isinstance(summary, dict):
        st.markdown(summary.get("beginner", "No summary available."))
    else:
        st.markdown(str(summary))

    st.markdown("---")

    # ML Model Comparison
    st.markdown("### 🤖 ML Model Predictions (5-Model Ensemble)")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🧪 Reproducibility Score by Model")
        model_preds = ml_repro.get("model_predictions", {})
        if model_preds:
            table_html = '<table class="model-table"><tr><th>Model</th><th>Score</th></tr>'
            for model, score in model_preds.items():
                table_html += f"<tr><td>{model}</td><td><strong>{score}/10</strong></td></tr>"
            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)
        st.markdown(f"**Interpretation:** {ml_repro.get('interpretation', 'N/A')}")

    with col_b:
        st.markdown("#### 📊 Difficulty Level by Model")
        model_preds_diff = ml_diff.get("model_predictions", {})
        if model_preds_diff:
            table_html = '<table class="model-table"><tr><th>Model</th><th>Prediction</th></tr>'
            for model, level in model_preds_diff.items():
                table_html += f"<tr><td>{model}</td><td><strong>{level}</strong></td></tr>"
            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)

        confidence = ml_diff.get("confidence", {})
        if confidence:
            st.markdown("**Confidence Scores:**")
            for level, pct in confidence.items():
                st.progress(pct / 100, text=f"{level}: {pct}%")

    st.markdown("")
    st.markdown("")

    # ── ML Outcome Summary ──
    st.markdown("### 💡 Model Outcome Summary")
    _repro_s = ml_repro.get("ensemble_score", "N/A")
    _diff_l = ml_diff.get("ensemble_level", "N/A")
    _interp = ml_repro.get("interpretation", "")
    _conf = ml_diff.get("confidence", {})
    _top_conf = max(_conf, key=_conf.get) if _conf else _diff_l
    _top_pct = _conf.get(_top_conf, 0) if _conf else 0

    outcome_lines = []
    if isinstance(_repro_s, (int, float)):
        if _repro_s >= 7:
            outcome_lines.append(f"This paper scores **{_repro_s}/10** on reproducibility — the methodology is well-documented and results should be straightforward to replicate.")
        elif _repro_s >= 4:
            outcome_lines.append(f"This paper scores **{_repro_s}/10** on reproducibility — some implementation details may need additional clarification before replication.")
        else:
            outcome_lines.append(f"This paper scores **{_repro_s}/10** on reproducibility — significant information gaps may make replication challenging without contacting the authors.")

    if _diff_l == "Beginner":
        outcome_lines.append(f"The difficulty is rated **{_diff_l}** ({_top_pct}% confidence), making it accessible for students and early-stage researchers.")
    elif _diff_l == "Intermediate":
        outcome_lines.append(f"The difficulty is rated **{_diff_l}** ({_top_pct}% confidence), requiring a solid foundation in the domain to fully understand.")
    elif _diff_l == "Advanced":
        outcome_lines.append(f"The difficulty is rated **{_diff_l}** ({_top_pct}% confidence), targeting experienced researchers with deep domain expertise.")

    outcome_lines.append("These predictions are generated by a 5-model ML ensemble (Random Forest, Gradient Boosting, Ridge/Logistic Regression, SVM, KNN) analyzing structural features of the paper.")

    st.markdown(
        f"""<div class="outcome-box">
            <span class="outcome-title">📋 What This Means For You</span>
            {'<br>'.join(outcome_lines)}
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Architecture Diagram
    st.markdown("### 🏗️ Architecture Diagram")
    diagram = analysis.get("diagram", "")
    if diagram and diagram != "Analysis temporarily unavailable. Please retry.":
        # Robust cleanup of the diagram text
        diagram_clean = diagram.strip()
        # Remove markdown fences
        diagram_clean = re.sub(r'```(?:mermaid)?\s*', '', diagram_clean)
        diagram_clean = re.sub(r'```\s*', '', diagram_clean)
        diagram_clean = diagram_clean.strip()
        
        # Convert unquoted square brackets: Node[Some text] -> Node["Some text"]
        diagram_clean = re.sub(r'(\b\w+\b)\[([^"\]]+)\]', r'\1["\2"]', diagram_clean)
        # Convert parentheses: Node(Some text) -> Node["Some text"]
        diagram_clean = re.sub(r'(\b\w+\b)\(([^)]+)\)', r'\1["\2"]', diagram_clean)
        # Convert curly braces: Node{Some text} -> Node["Some text"]
        diagram_clean = re.sub(r'(\b\w+\b)\{([^}]+)\}', r'\1["\2"]', diagram_clean)
        
        # Ensure it starts with graph or flowchart
        if not diagram_clean.startswith(('graph ', 'flowchart ')):
            diagram_clean = 'graph TD\n' + diagram_clean

        mermaid_html = f"""
        <div style="background:rgba(22,33,62,0.75);border:1px solid rgba(139,92,246,0.25);border-radius:16px;padding:24px;margin:12px 0;">
            <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
            <script>
                mermaid.initialize({{ startOnLoad: true, theme: 'dark', themeVariables: {{ primaryColor: '#8B5CF6', primaryTextColor: '#F1F5F9', lineColor: '#A78BFA', secondaryColor: '#1E1B4B' }} }});
            </script>
            <pre class="mermaid">
{diagram_clean}
            </pre>
        </div>
        """
        st.components.v1.html(mermaid_html, height=500, scrolling=True)
    else:
        st.info("Architecture diagram not available.")

    st.markdown("---")

    # AI Reproducibility Analysis
    st.markdown("### 🔬 AI Reproducibility Analysis")
    st.markdown(analysis.get("reproducibility_analysis", "Not available."))


# ── TAB 2: Math & Concepts ──
with tab2:
    st.markdown("## ➗ Mathematical Insights")
    st.markdown(analysis.get("math_explanation", "No math analysis available."))

    st.markdown("---")

    st.markdown("## 🧩 Knowledge Gap Detector")
    st.markdown(analysis.get("knowledge_gaps", "No knowledge gap analysis available."))


# ── TAB 3: Innovation ──
with tab3:
    st.markdown("## 🔬 Innovation Difference Engine")
    st.markdown(analysis.get("innovation_analysis", "No innovation analysis available."))


# ── TAB 4: Implementation ──
with tab4:
    st.markdown("## 🛠️ Implementation Roadmap")
    st.markdown(analysis.get("roadmap", "No roadmap available."))

    st.markdown("---")

    st.markdown("## 💻 Simplified Starter Code")
    code = analysis.get("code", "")
    if code and code != "Analysis temporarily unavailable. Please retry.":
        st.markdown(code)
    else:
        st.info("Starter code not available.")


# ── TAB 5: Interactive Quiz ──
with tab5:
    st.markdown("## 📝 AI-Generated Quiz")

    quiz_raw = analysis.get("quiz", "")

    # Try to parse as JSON for interactive MCQ
    quiz_data = None
    if quiz_raw and quiz_raw != "Analysis temporarily unavailable. Please retry.":
        try:
            # Try direct JSON parse
            quiz_data = json.loads(quiz_raw)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown fences
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', quiz_raw, re.DOTALL)
            if json_match:
                try:
                    quiz_data = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            if not quiz_data:
                # Try to find JSON object in the text
                brace_match = re.search(r'\{[\s\S]*\}', quiz_raw)
                if brace_match:
                    try:
                        quiz_data = json.loads(brace_match.group(0))
                    except json.JSONDecodeError:
                        pass

    if quiz_data and "mcq" in quiz_data:
        mcqs = quiz_data["mcq"]

        # Initialize quiz state
        if "quiz_answers" not in st.session_state:
            st.session_state.quiz_answers = {}
        if "quiz_submitted" not in st.session_state:
            st.session_state.quiz_submitted = {}

        st.markdown("### 🎯 Multiple Choice Questions")
        st.markdown("*Select your answer and click Check to see if you're correct!*")
        st.markdown("")

        score = 0
        total_answered = 0

        for i, q in enumerate(mcqs):
            qkey = f"quiz_q_{i}"
            skey = f"quiz_s_{i}"

            st.markdown(
                f'<div class="quiz-question-card">'
                f'<span class="quiz-question-num">QUESTION {i+1}</span>'
                f'<div class="quiz-question-text">{q.get("question", "")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            options = q.get("options", {})
            option_list = [f"{k}) {v}" for k, v in options.items()]

            selected = st.radio(
                f"Select answer for Q{i+1}:",
                option_list,
                key=qkey,
                label_visibility="collapsed",
            )

            col_check, col_spacer = st.columns([1, 4])
            with col_check:
                if st.button(f"✅ Check Answer", key=f"btn_{i}"):
                    st.session_state.quiz_submitted[skey] = True

            if st.session_state.quiz_submitted.get(skey):
                user_letter = selected[0] if selected else ""
                correct_letter = q.get("correct", "")
                explanation = q.get("explanation", "")
                total_answered += 1

                if user_letter == correct_letter:
                    score += 1
                    st.markdown(
                        f'<div class="quiz-result-correct">✅ <strong>Correct!</strong> You selected {correct_letter}.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    correct_text = options.get(correct_letter, "")
                    st.markdown(
                        f'<div class="quiz-result-incorrect">❌ <strong>Incorrect.</strong> You selected {user_letter}. The correct answer is <strong>{correct_letter}) {correct_text}</strong></div>',
                        unsafe_allow_html=True,
                    )

                if explanation:
                    st.markdown(
                        f'<div class="quiz-explanation">💡 <strong>Explanation:</strong> {explanation}</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("")

        # Score summary
        if total_answered == len(mcqs):
            pct = int((score / len(mcqs)) * 100)
            emoji = "🏆" if pct >= 80 else "👍" if pct >= 60 else "📚"
            st.markdown(
                f'<div class="quiz-score-card">'
                f'<div class="quiz-score-value">{emoji} {score}/{len(mcqs)}</div>'
                f'<div class="quiz-score-label">Your Score — {pct}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Other question types
        st.markdown("---")

        for section_key, section_title, section_emoji in [
            ("conceptual", "Conceptual Questions", "💭"),
            ("coding", "Coding Questions", "💻"),
            ("viva", "Viva Questions", "🎤"),
        ]:
            items = quiz_data.get(section_key, [])
            if items:
                st.markdown(f"### {section_emoji} {section_title}")
                for j, item in enumerate(items):
                    with st.expander(f"Q{j+1}: {item.get('question', '')[:80]}..."):
                        st.markdown(f"**Question:** {item.get('question', '')}")
                        if item.get("hint"):
                            st.markdown(f"💡 **Hint:** {item['hint']}")
                st.markdown("")

    else:
        # Fallback: show raw quiz text
        if quiz_raw:
            st.markdown(quiz_raw)
        else:
            st.info("No quiz available.")


# ── TAB 6: Chat ──
with tab6:
    st.markdown("## 💬 Chat with Your Paper")
    st.markdown("*Ask anything about the paper — powered by RAG + Llama 3.1*")
    st.markdown("---")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about the paper... e.g., 'Explain section 3 simply'"):
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/chat",
                        json={"paper_id": st.session_state.paper_id, "question": prompt},
                        timeout=120,
                    )
                    if response.status_code == 200:
                        answer = response.json()["answer"]
                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend.")
                except Exception as e:
                    st.error(f"Error: {e}")
