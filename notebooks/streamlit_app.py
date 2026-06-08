"""
ACME Industries — Streamlit Predictive Maintenance Dashboard
=============================================================
Interactive web dashboard for real-time ML/AI predictive maintenance.

Run:
    streamlit run streamlit_app.py --server.port 8501

Requires FastAPI backend running on port 8000, or falls back to direct import.
"""

import os
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import streamlit as st
import requests
import json
import numpy as np

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="ACME Industries — Predictive Maintenance HQ",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS for Premium Dark Industrial Theme ────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global Theme ── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 40%, #2d1b69 70%, #1a1a3e 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -30%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-banner h1 {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        color: rgba(255,255,255,0.6);
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
        font-weight: 400;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #16162a 100%);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    .metric-label {
        color: rgba(255,255,255,0.5);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }
    .metric-subtitle {
        color: rgba(255,255,255,0.4);
        font-size: 0.8rem;
        margin-top: 0.3rem;
    }

    /* ── Status Badges ── */
    .status-critical {
        background: linear-gradient(135deg, #dc2626, #991b1b);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 10px rgba(220, 38, 38, 0.3);
    }
    .status-warning {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: #1a1a2e;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 10px rgba(245, 158, 11, 0.3);
    }
    .status-normal {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 10px rgba(16, 185, 129, 0.3);
    }

    /* ── Info Cards ── */
    .info-card {
        background: linear-gradient(145deg, #1e1e2e, #16162a);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 0.8rem;
    }
    .info-card h4 {
        color: #818cf8;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.6rem;
    }
    .info-card p {
        color: rgba(255,255,255,0.8);
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 0;
    }

    /* ── Sidebar Styling ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #818cf8;
        font-size: 1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* ── Tab Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* ── Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }

    /* ── Hide default Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── API Configuration ───────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


def call_api(endpoint, method="GET", payload=None):
    """Make HTTP request to FastAPI backend with fallback."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "POST":
            resp = requests.post(url, json=payload, timeout=30)
        else:
            resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "API_UNREACHABLE"
    except Exception as e:
        return None, str(e)


def call_engine_directly(payload, operator_log, machine_id):
    """Fallback: import core_engine directly if API is down."""
    try:
        import core_engine
        raw_sensors = {k: v for k, v in payload.items() if k not in ["operator_log", "machine_id"]}
        ticket = core_engine.master_orchestrator(raw_sensors, operator_log, machine_id)
        
        # Append API contract keys at the top-level
        prob_raw = ticket.get("Telemetry_Metrics", {}).get("Failure_Probability_Raw", 0.0)
        prediction = "Failure" if prob_raw >= 0.50 else "No Failure"
        confidence = float(prob_raw)
        
        failure_type = None
        if prediction == "Failure":
            failure_type = ticket.get("Root_Cause_Diagnosis", {}).get("Failure_Mode", "UNKNOWN")

        ticket["prediction"] = prediction
        ticket["confidence"] = confidence
        ticket["failure_type"] = failure_type
        
        return ticket, None
    except Exception as e:
        return None, str(e)


# ─── Feature Names & Defaults ────────────────────────────────────────────────
FEATURE_NAMES = [
    "torque_nm_freq_dev", "power_w_freq_dev", "power_w_rms_dev", "torque_nm_rms_dev",
    "feat_stat_std", "rotational_speed_rpm_rms_dev", "feat_stat_range", "rotational_speed_rpm_freq_dev",
    "power_speed_ratio", "wear_torque_interact", "mech_stress", "feat_stat_mean",
    "temp_delta_k_rms_dev", "power_w_roll10_max", "temp_delta_k_roll10_mean", "temp_delta_k_roll10_max",
    "temp_delta_k", "torque_nm_roll10_max", "temp_delta_k_freq_dev", "tool_wear_ratio"
]

# Friendly display names for key features
FEATURE_DISPLAY = {
    "torque_nm_freq_dev": "⚙️ Torque Frequency Deviation",
    "mech_stress": "🔩 Mechanical Stress",
    "tool_wear_ratio": "🛠️ Tool Wear Ratio",
    "rotational_speed_rpm_rms_dev": "🔄 Rotational Speed RMS Dev",
    "power_w_freq_dev": "⚡ Power Frequency Deviation",
    "wear_torque_interact": "🔗 Wear-Torque Interaction",
    "temp_delta_k": "🌡️ Temperature Delta",
    "feat_stat_std": "📊 Feature Std Deviation",
}

# Primary slider features (most important for user interaction)
PRIMARY_FEATURES = ["torque_nm_freq_dev", "mech_stress", "tool_wear_ratio", "rotational_speed_rpm_rms_dev"]


# ══════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-banner">
    <h1>🏭 ACME Industries — Predictive Maintenance HQ</h1>
    <p>Enterprise AI-Powered Failure Prediction • Anomaly Detection • Root Cause Diagnosis</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Input Controls
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎛️ Sensor Controls")
    st.markdown("---")

    # ── Raw Sensor Inputs ──
    st.markdown("##### Raw Telemetry Inputs")
    torque_nm = st.number_input("⚙️ Torque (Nm)", min_value=0.0, max_value=100.0, value=40.0, step=0.1, key="torque_nm")
    spindle_speed_rpm = st.number_input("🔩 Spindle Speed (RPM)", min_value=0.0, max_value=5000.0, value=1500.0, step=50.0, key="spindle_speed_rpm")
    tool_wear_min = st.slider("🛠️ Tool Wear (min)", min_value=0.0, max_value=250.0, value=25.0, step=1.0, key="tool_wear_min")
    rotational_speed_rpm = st.number_input("🔄 Rotational Speed (RPM)", min_value=0.0, max_value=5000.0, value=1500.0, step=50.0, key="rotational_speed_rpm")
    power_w = st.number_input("⚡ Power (W)", min_value=0.0, max_value=20000.0, value=6000.0, step=100.0, key="power_w")
    voltage_v = st.number_input("🔌 Voltage (V)", min_value=0.0, max_value=500.0, value=220.0, step=1.0, key="voltage_v")
    current_a = st.number_input("🔋 Current (A)", min_value=0.0, max_value=100.0, value=28.0, step=0.5, key="current_a")
    vibration_mm_s = st.number_input("📳 Vibration Amplitude (mm/s)", min_value=0.0, max_value=20.0, value=1.5, step=0.1, key="vibration_mm_s")
    temperature_k = st.number_input("🌡️ Temperature (K)", min_value=200.0, max_value=400.0, value=310.0, step=0.1, key="temperature_k")


    st.markdown("---")
    st.markdown("##### 📝 Operator Log")
    operator_log = st.text_area(
        "Describe the observed issue:",
        value="Main conveyor layout stalled. Inspection reveals complete mechanical fracture affecting the drive belt.",
        height=120,
        key="operator_log"
    )

    st.markdown("---")
    machine_id = st.text_input("🏷️ Machine ID", value="M-1042", key="machine_id")

    st.markdown("---")
    analyze_btn = st.button("🔴 ANALYZE TELEMETRY", key="analyze_button", use_container_width=True)

    # ── Connection Status ──
    st.markdown("---")
    st.markdown("##### 📡 API Status")
    health_data, health_err = call_api("/health")
    if health_data:
        status_color = "🟢" if health_data.get("status") == "healthy" else "🟡"
        st.markdown(f"{status_color} **{health_data.get('status', 'unknown').upper()}**")
        st.caption(f"LightGBM: {'✅' if health_data.get('failure_model_loaded') else '❌'} | "
                   f"IsoForest: {'✅' if health_data.get('anomaly_model_loaded') else '❌'} | "
                   f"NLP: {'✅' if health_data.get('nlp_model_loaded') else '❌'}")
    else:
        st.markdown("🟠 **DIRECT MODE** (API offline)")
        st.caption("Using direct engine import as fallback")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT — Tabbed Interface
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["📊 Live Diagnostics", "📋 Dispatch Ticket", "📈 Model Performance"])

# ─── Run Analysis ────────────────────────────────────────────────────────────
if analyze_btn:
    with st.spinner("⚡ Running ML/AI inference pipeline..."):
        payload = {
            "torque_nm": float(torque_nm),
            "spindle_speed_rpm": float(spindle_speed_rpm),
            "tool_wear_min": float(tool_wear_min),
            "rotational_speed_rpm": float(rotational_speed_rpm),
            "power_w": float(power_w),
            "voltage_v": float(voltage_v),
            "current_a": float(current_a),
            "vibration_mm_s": float(vibration_mm_s),
            "temperature_k": float(temperature_k),
            "operator_log": operator_log,
            "machine_id": machine_id
        }


        # Try API first, fallback to direct engine
        result, error = call_api("/predict", method="POST", payload=payload)
        if error == "API_UNREACHABLE":
            result, error = call_engine_directly(payload, operator_log, machine_id)

        if error:
            st.error(f"❌ Pipeline Error: {error}")
        else:
            st.session_state["last_result"] = result
            st.session_state["last_payload"] = payload

# ── Retrieve last result ──
result = st.session_state.get("last_result", None)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: LIVE DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    if result is None:
        st.info("👈 Configure sensor values in the sidebar and click **ANALYZE TELEMETRY** to begin.")
    else:
        status = result.get("Orchestration_Status", "UNKNOWN")
        telemetry = result.get("Telemetry_Metrics", {})
        anomaly = result.get("Anomaly_Detection", {})
        xai = result.get("Explainable_AI_Insight", {})
        diagnosis = result.get("Root_Cause_Diagnosis", {})
        plan = result.get("Actionable_Maintenance_Plan", {})

        failure_prob_raw = telemetry.get("Failure_Probability_Raw", 0)
        if failure_prob_raw is None:
            failure_prob_raw = 0
        priority = telemetry.get("System_Priority", "NORMAL")

        # ── Row 1: Key Metrics ──
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            priority_class = {
                "CRITICAL": "status-critical",
                "WARNING": "status-warning",
                "NORMAL": "status-normal"
            }.get(priority, "status-normal")

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">System Priority</div>
                <span class="{priority_class}">{priority}</span>
                <div class="metric-subtitle">Ticket: {result.get('Ticket_ID', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            prob_color = "#dc2626" if failure_prob_raw >= 0.8 else ("#f59e0b" if failure_prob_raw >= 0.5 else "#10b981")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Failure Probability</div>
                <p class="metric-value" style="color: {prob_color};">{telemetry.get('Failure_Probability', 'N/A')}</p>
                <div class="metric-subtitle">LightGBM Prediction</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            anom_status = anomaly.get("Status", "N/A")
            anom_color = "#dc2626" if "ANOMALY" in str(anom_status) else "#10b981"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Anomaly Detection</div>
                <p class="metric-value" style="color: {anom_color}; font-size: 1.3rem;">{anom_status}</p>
                <div class="metric-subtitle">Score: {anomaly.get('Anomaly_Score', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Dispatch Status</div>
                <p class="metric-value" style="color: {'#dc2626' if status == 'DISPATCHED' else '#10b981'}; font-size: 1.3rem;">{status}</p>
                <div class="metric-subtitle">Machine: {result.get('Machine_ID', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Progress Bar ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(min(float(failure_prob_raw), 1.0), text=f"Risk Level: {float(failure_prob_raw)*100:.1f}%")

        # ── Row 2: XAI + Diagnosis ──
        if status == "DISPATCHED":
            st.markdown("<br>", unsafe_allow_html=True)
            col_left, col_right = st.columns(2)

            with col_left:
                # XAI Card
                risk_drivers = xai.get("Risk_Drivers", [])
                drivers_html = "".join([f"<li style='color:rgba(255,255,255,0.8);margin:4px 0;'>🔸 {d.replace('_', ' ').title()}</li>" for d in risk_drivers])
                st.markdown(f"""
                <div class="info-card">
                    <h4>🧠 Explainable AI — Top Risk Factors</h4>
                    <ul style="list-style:none;padding-left:0;">{drivers_html}</ul>
                    <p style="margin-top:0.8rem;font-style:italic;color:rgba(255,255,255,0.6);font-size:0.85rem;">
                        {xai.get('Generated_Explanation', '')}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Diagnosis Card
                st.markdown(f"""
                <div class="info-card">
                    <h4>🔍 Root Cause Diagnosis</h4>
                    <p><strong>Component:</strong> {diagnosis.get('Affected_Component', 'N/A')}</p>
                    <p><strong>Failure Mode:</strong> {diagnosis.get('Failure_Mode', 'N/A')}</p>
                    <p><strong>Method:</strong> {diagnosis.get('Diagnosis_Method', 'N/A')}</p>
                    <p><strong>Confidence:</strong> {diagnosis.get('Match_Confidence', 'N/A')}</p>
                    <p><strong>Reference:</strong> {diagnosis.get('Source_Context_Reference', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)

            with col_right:
                # Maintenance Plan Card
                st.markdown(f"""
                <div class="info-card">
                    <h4>🔧 Actionable Maintenance Plan</h4>
                    <p><strong>Recommended Fix:</strong></p>
                    <p>{plan.get('Recommended_Fix', 'N/A')}</p>
                    <p style="margin-top:0.8rem;"><strong>Required Inventory:</strong></p>
                    <p style="color:#818cf8;font-weight:600;">{plan.get('Required_Inventory', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)

                # Operator Log Card
                st.markdown(f"""
                <div class="info-card">
                    <h4>📝 Ingested Operator Log</h4>
                    <p>{result.get('Ingested_Operator_Log', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)

        elif status == "RESOLVED":
            st.success(f"✅ {result.get('Message', 'Machine metrics within normal bounds.')}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: DISPATCH TICKET (JSON)
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    if result is None:
        st.info("No dispatch ticket generated yet. Run an analysis first.")
    else:
        st.markdown("### 📋 Unified Dispatch Payload")

        # Status banner
        status = result.get("Orchestration_Status", "UNKNOWN")
        if status == "DISPATCHED":
            st.error(f"🚨 CRITICAL DISPATCH — Ticket {result.get('Ticket_ID', 'N/A')}")
        else:
            st.success(f"✅ RESOLVED — {result.get('Ticket_ID', 'N/A')}")

        # Full JSON
        st.json(result)

        # Download button
        json_str = json.dumps(result, indent=4)
        st.download_button(
            label="📥 Download Ticket JSON",
            data=json_str,
            file_name=f"{result.get('Ticket_ID', 'ticket')}.json",
            mime="application/json"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### 📈 Model Performance Dashboard")
    st.caption("Live metrics computed from saved test splits (models/*.npy)")

    # Try to get performance data
    perf_data, perf_err = call_api("/models/performance")

    if perf_err == "API_UNREACHABLE":
        # Fallback: compute directly
        try:
            import core_engine
            perf_data = core_engine.get_model_performance()
        except Exception as e:
            perf_data = None
            st.warning(f"Could not load performance data: {e}")

    if perf_data:
        # ── Tier 1: LightGBM ──
        st.markdown("---")
        lgbm = perf_data.get("lightgbm_failure_predictor", {})
        lgbm_metrics = lgbm.get("metrics", {})

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tier 1: Failure Predictor</div>
                <p class="metric-value" style="color:#6366f1;font-size:1.4rem;">{lgbm.get('model_name', 'N/A')}</p>
                <div class="metric-subtitle">Status: {lgbm.get('status', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if isinstance(lgbm_metrics.get("auc_roc"), (int, float)):
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("AUC-ROC", f"{lgbm_metrics['auc_roc']:.4f}")
                mcol2.metric("F1 Score", f"{lgbm_metrics['f1_score']:.4f}")
                mcol3.metric("Precision", f"{lgbm_metrics['precision']:.4f}")
                mcol4.metric("Recall", f"{lgbm_metrics['recall']:.4f}")

                # Classification report
                cr = lgbm.get("classification_report", {})
                if cr:
                    with st.expander("📊 Full Classification Report"):
                        import pandas as pd
                        cr_df = pd.DataFrame(cr).T
                        st.dataframe(cr_df.style.format("{:.4f}", na_rep="-"), use_container_width=True)
            else:
                st.warning("Model is using mock fallback — no real metrics available.")

        # ── Tier 2: Isolation Forest ──
        st.markdown("---")
        iso = perf_data.get("isolation_forest_anomaly", {})
        iso_metrics = iso.get("metrics", {})

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tier 2: Anomaly Detector</div>
                <p class="metric-value" style="color:#f59e0b;font-size:1.4rem;">{iso.get('model_name', 'N/A')}</p>
                <div class="metric-subtitle">Status: {iso.get('status', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if isinstance(iso_metrics.get("auc_roc"), (int, float)):
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("AUC-ROC", f"{iso_metrics['auc_roc']:.4f}")
                mcol2.metric("F1 Score", f"{iso_metrics['f1_score']:.4f}")
                mcol3.metric("Precision", f"{iso_metrics['precision']:.4f}")
                mcol4.metric("Recall", f"{iso_metrics['recall']:.4f}")
            else:
                st.warning("Model is using mock fallback — no real metrics available.")

        # ── Tier 3: NLP Agent ──
        st.markdown("---")
        nlp = perf_data.get("nlp_diagnosis_agent", {})
        hist = nlp.get("historical_evaluation", {})

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tier 3: NLP Diagnosis</div>
                <p class="metric-value" style="color:#10b981;font-size:1.1rem;">BERT + TF-IDF</p>
                <div class="metric-subtitle">Status: {nlp.get('status', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Component Accuracy", hist.get("component_extraction_accuracy", "N/A"))
            mcol2.metric("Failure Mode Accuracy", hist.get("failure_mode_accuracy", "N/A"))
            mcol3.metric("Eval Dataset Size", hist.get("evaluation_dataset_size", "N/A"))
            st.caption(f"Method: {nlp.get('diagnosis_method', 'N/A')} | Corpus Size: {nlp.get('rag_corpus_size', 'N/A')} documents")

        # ── System Summary ──
        st.markdown("---")
        summary = perf_data.get("system_summary", {})
        st.markdown(f"""
        <div class="info-card">
            <h4>🖥️ System Summary</h4>
            <p><strong>Platform:</strong> {summary.get('platform', 'N/A')} |
               <strong>Models Loaded:</strong> {summary.get('models_loaded', 0)}/{summary.get('total_models', 3)} |
               <strong>Features:</strong> {summary.get('feature_count', 0)}</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("Unable to load model performance data. Ensure either the API is running or core_engine is accessible.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:rgba(255,255,255,0.3);font-size:0.75rem;'>"
    "ACME Industries © 2024 — Predictive Maintenance Platform v2.0 | "
    "Powered by LightGBM • Isolation Forest • HuggingFace BERT • TF-IDF RAG"
    "</div>",
    unsafe_allow_html=True
)
