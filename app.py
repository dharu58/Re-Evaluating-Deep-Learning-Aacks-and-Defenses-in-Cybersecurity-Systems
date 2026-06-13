import streamlit as st
import json
import plotly.graph_objects as go
import plotly.express as px
import os

# ─────────────────────────────────────────────
# 1. Page Config & CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title="IDS Adversarial Robustness Dashboard", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #080c14;
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] {
        background-color: #0d1220;
        border-right: 1px solid #1e2a40;
    }
    .sidebar-title {
        font-family: 'Space Mono', monospace;
        font-size: 13px;
        font-weight: 700;
        color: #38bdf8;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .sidebar-subtitle {
        font-size: 11px;
        color: #64748b;
        margin-bottom: 20px;
        line-height: 1.5;
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #0d1220 0%, #111827 100%);
        border: 1px solid #1e2a40;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: var(--accent, #38bdf8);
    }
    .kpi-label {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-family: 'Space Mono', monospace;
        font-size: 32px;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1;
    }
    .kpi-delta {
        font-size: 12px;
        margin-top: 6px;
    }
    .kpi-delta.neg { color: #f87171; }
    .kpi-delta.pos { color: #4ade80; }
    .kpi-delta.neu { color: #94a3b8; }

    /* Threat Callout Cards */
    .threat-card {
        background: #0d1220;
        border: 1px solid #1e2a40;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .threat-tag {
        display: inline-block;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .tag-danger  { background:#7f1d1d; color:#fca5a5; }
    .tag-warning { background:#78350f; color:#fcd34d; }
    .tag-success { background:#14532d; color:#86efac; }

    .threat-name {
        font-family: 'Space Mono', monospace;
        font-size: 22px;
        font-weight: 700;
        color: #f8fafc;
    }
    .threat-desc {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Section header */
    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        color: #38bdf8;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        border-bottom: 1px solid #1e2a40;
        padding-bottom: 8px;
        margin: 28px 0 16px 0;
    }

    /* Page title */
    .page-title {
        font-family: 'Space Mono', monospace;
        font-size: 26px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .page-subtitle {
        font-size: 13px;
        color: #64748b;
        margin-bottom: 28px;
    }

    /* Clean table */
    .stDataFrame, .stTable { background: #0d1220 !important; }

    div[data-testid="stMetric"] {
        background: #0d1220;
        border: 1px solid #1e2a40;
        border-radius: 10px;
        padding: 16px;
    }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 2. Data Loading
# ─────────────────────────────────────────────
@st.cache_data
def load_all_data():
    attack_data = {}
    files = {
        "GAN": "gan.json",
        "KDE": "kde.json",
        "DeepFool": "deepfool.json",
        "ZOO": "zoo.json"
    }
    for name, filename in files.items():
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                attack_data[name] = json.load(f)
        else:
            st.error(f"Missing data file: {filename}")
    return attack_data

data = load_all_data()

COLORS = {
    "GAN":      "#38bdf8",   # sky blue
    "KDE":      "#a78bfa",   # violet
    "DeepFool": "#f87171",   # red
    "ZOO":      "#4ade80",   # green
}

def hex_to_rgba(hex_color, alpha=1.0):
    """Convert a hex color string to rgba() format Plotly accepts."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#94a3b8', size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1e2a40', borderwidth=1),
)

def axis_style(title=""):
    return dict(
        title=title,
        title_font=dict(color='#64748b', size=11),
        tickfont=dict(color='#64748b'),
        showgrid=True,
        gridcolor='#1e2a40',
        zeroline=False,
    )


# ─────────────────────────────────────────────
# 3. Sidebar
# ─────────────────────────────────────────────
st.sidebar.markdown('<div class="sidebar-title">⬡ IDS Adversarial Lab</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-subtitle">CICIDS2018 · Adversarial Robustness Evaluation</div>', unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", ["Overview", "Attack Simulation", "Detection Analysis", "Defense Analysis"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Configuration**")

dataset = st.sidebar.selectbox("Dataset", ["CICIDS2018"])

available_attacks = list(data.keys())
selected_attacks = st.sidebar.multiselect("Select Attack Vectors", available_attacks, default=available_attacks)

attack_ratio = st.sidebar.slider("Attack Ratio (%)", min_value=1, max_value=5, value=5)

st.sidebar.markdown("---")
st.sidebar.markdown('<span style="font-size:11px;color:#334155;">Final Year Project · BTP</span>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper: KPI card
# ─────────────────────────────────────────────
def kpi(label, value, delta=None, delta_type="neu", accent="#38bdf8"):
    delta_html = ""
    if delta is not None:
        delta_html = f'<div class="kpi-delta {delta_type}">{delta}</div>'
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════
if page == "Overview":
    st.markdown('<div class="page-title">🛡 Adversarial Robustness Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">IDS evaluation under adversarial attack vectors · CICIDS2018 dataset</div>', unsafe_allow_html=True)

    if not selected_attacks:
        st.warning("Select at least one attack vector from the sidebar.")
    else:
        # ── Baseline Accuracy KPIs ───────────────────
        st.markdown('<div class="section-header">Baseline Accuracy by Attack Vector</div>', unsafe_allow_html=True)
        cols = st.columns(len(selected_attacks))
        for idx, atk in enumerate(selected_attacks):
            with cols[idx]:
                kpi(f"{atk} Baseline", f"{data[atk]['baseline_accuracy']}%", accent=COLORS.get(atk, "#38bdf8"))

        # ── Grouped Horizontal Bar Comparison ────────
        st.markdown('<div class="section-header">Attack Comparison — Baseline vs Post-Attack vs Defense</div>', unsafe_allow_html=True)

        metrics = ['Baseline', 'Post-Attack', 'Defense']
        fig_comp = go.Figure()

        for atk in selected_attacks:
            atk_res = next((r for r in data[atk]['attack_results'] if r['attack_percentage'] == attack_ratio), None)
            post_acc = atk_res['accuracy_after_attack'] if atk_res else data[atk]['baseline_accuracy']
            defr_list = data[atk].get('defense_results', {}).get('after_attack_defense', [])
            def_entry = next((e for e in defr_list if e.get('attack_percentage') == attack_ratio), None)
            def_acc = def_entry.get('defense_accuracy', data[atk]['baseline_accuracy']) if def_entry else data[atk]['baseline_accuracy']

            vals = [data[atk]['baseline_accuracy'], post_acc, def_acc]
            color = COLORS.get(atk, "#38bdf8")

            fig_comp.add_trace(go.Bar(
                name=atk,
                x=vals,
                y=metrics,
                orientation='h',
                marker_color=color,
                text=[f"{v:.1f}%" for v in vals],
                textposition='inside',
                textfont=dict(color='#0d1220', size=11, family='Space Mono'),
            ))

        fig_comp.update_layout(
            **PLOT_LAYOUT,
            barmode='group',
            xaxis={**axis_style("Accuracy (%)"), "range": [75, 105]},
            yaxis=dict(tickfont=dict(color='#94a3b8', size=12), showgrid=False, zeroline=False),
            height=360,
            hovermode="y unified",
        )
        st.plotly_chart(fig_comp, use_container_width=True)


# ═══════════════════════════════════════════
# PAGE 2: ATTACK SIMULATION
# ═══════════════════════════════════════════
elif page == "Attack Simulation":
    st.markdown('<div class="page-title">⚡ Attack Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Accuracy degradation under increasing adversarial pressure</div>', unsafe_allow_html=True)

    if not selected_attacks:
        st.warning("Select at least one attack vector from the sidebar.")
    else:
        # KPIs
        avg_baseline = sum(data[atk]['baseline_accuracy'] for atk in selected_attacks) / len(selected_attacks)
        post_accs = []
        for atk in selected_attacks:
            res = next((r for r in data[atk]['attack_results'] if r['attack_percentage'] == attack_ratio), None)
            post_accs.append(res['accuracy_after_attack'] if res else data[atk]['baseline_accuracy'])
        avg_after = sum(post_accs) / len(post_accs)
        avg_drop = avg_baseline - avg_after

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi("Avg Baseline Accuracy", f"{avg_baseline:.1f}%", accent="#38bdf8")
        with c2:
            kpi("Avg After Attack", f"{avg_after:.1f}%",
                delta=f"↓ {avg_drop:.2f}% drop at {attack_ratio}% ratio", delta_type="neg", accent="#f87171")
        with c3:
            worst_atk = selected_attacks[post_accs.index(min(post_accs))]
            kpi("Most Impacted", worst_atk,
                delta=f"{min(post_accs):.1f}% post-attack accuracy", delta_type="neg", accent="#f87171")

        # ── Accuracy Trend Lines ─────────────────────
        st.markdown('<div class="section-header">Accuracy Degradation Curve (1–5% Attack Ratio)</div>', unsafe_allow_html=True)

        fig_trend = go.Figure()
        ratios = [1, 2, 3, 4, 5]
        for atk in selected_attacks:
            y_vals = []
            for r in ratios:
                entry = next((x for x in data[atk]['attack_results'] if x['attack_percentage'] == r), None)
                y_vals.append(entry['accuracy_after_attack'] if entry else data[atk]['baseline_accuracy'])
            fig_trend.add_trace(go.Scatter(
                x=ratios, y=y_vals, name=atk, mode='lines+markers',
                line=dict(color=COLORS.get(atk, "#38bdf8"), width=2.5),
                marker=dict(size=7, color=COLORS.get(atk, "#38bdf8")),
            ))
            # Baseline dotted reference line
            fig_trend.add_trace(go.Scatter(
                x=[1, 5], y=[data[atk]['baseline_accuracy']] * 2,
                mode='lines', name=f"{atk} Baseline",
                line=dict(color=COLORS.get(atk, "#38bdf8"), width=1, dash='dot'),
                showlegend=False, opacity=0.4,
            ))

        # Highlight selected ratio
        fig_trend.add_vline(x=attack_ratio, line_dash="dash", line_color="#475569",
                            annotation_text=f"  Selected: {attack_ratio}%",
                            annotation_font=dict(color="#64748b", size=11))

        fig_trend.update_layout(
            **PLOT_LAYOUT,
            xaxis={**axis_style("Attack Ratio (%)"), "tickvals": ratios},
            yaxis={**axis_style("Accuracy (%)"), "range": [80, 102]},
            height=420,
            hovermode="x unified",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # ── Bar Chart at Selected Ratio ──────────────
        st.markdown(f'<div class="section-header">Attack Impact at {attack_ratio}% Ratio — Grouped Comparison</div>', unsafe_allow_html=True)

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Baseline",
            x=selected_attacks,
            y=[data[atk]['baseline_accuracy'] for atk in selected_attacks],
            marker_color=[hex_to_rgba(COLORS.get(atk, "#38bdf8"), 0.3) for atk in selected_attacks],
            marker_line_color=[COLORS.get(atk, "#38bdf8") for atk in selected_attacks],
            marker_line_width=1.5,
        ))
        fig_bar.add_trace(go.Bar(
            name="After Attack",
            x=selected_attacks,
            y=post_accs,
            marker_color=[COLORS.get(atk, "#38bdf8") for atk in selected_attacks],
            text=[f"{v:.1f}%" for v in post_accs],
            textposition='auto',
            textfont=dict(color='#0d1220', size=11),
        ))

        fig_bar.update_layout(
            **PLOT_LAYOUT,
            barmode='group',
            xaxis=axis_style("Attack Vector"),
            yaxis={**axis_style("Accuracy (%)"), "range": [80, 102]},
            height=380,
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════════════════════
# PAGE 3: DETECTION ANALYSIS
# ═══════════════════════════════════════════
elif page == "Detection Analysis":
    st.markdown('<div class="page-title">🔍 Detection Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Evaluating adversarial sample identification before pipeline compromise</div>', unsafe_allow_html=True)

    if not selected_attacks:
        st.warning("Select at least one attack vector from the sidebar.")
    else:
        det_accs = []
        for atk in selected_attacks:
            if 'detection_results' in data[atk]:
                dr = data[atk]['detection_results']
                val = dr.get('attack_detection_accuracy', dr.get('overall_detection_accuracy', 0))
                det_accs.append(val)
        avg_det = sum(det_accs) / len(det_accs) if det_accs else 0

        # ── Detection Accuracy Bar ───────────────────
        st.markdown('<div class="section-header">Attack Detection Accuracy by Attack Vector</div>', unsafe_allow_html=True)

        det_x, det_y, det_colors = [], [], []
        for atk in selected_attacks:
            if 'detection_results' in data[atk]:
                dr = data[atk]['detection_results']
                det_x.append(atk)
                det_y.append(dr.get('attack_detection_accuracy', dr.get('overall_detection_accuracy', 0)))
                det_colors.append(COLORS.get(atk, "#38bdf8"))

        fig_det = go.Figure(go.Bar(
            x=det_x, y=det_y,
            marker_color=det_colors,
            text=[f"{v:.1f}%" for v in det_y],
            textposition='auto',
            textfont=dict(color='#0d1220', size=11),
            width=0.5,
        ))
        # Reference line at 100%
        fig_det.add_hline(y=100, line_dash="dot", line_color="#334155",
                          annotation_text="100% threshold", annotation_font=dict(color="#475569", size=10))

        fig_det.update_layout(
            **PLOT_LAYOUT,
            xaxis=axis_style("Attack Vector"),
            yaxis={**axis_style("Detection Accuracy (%)"), "range": [0, 110]},
            height=380,
        )
        st.plotly_chart(fig_det, use_container_width=True)




# ═══════════════════════════════════════════
# PAGE 4: DEFENSE ANALYSIS
# ═══════════════════════════════════════════
elif page == "Defense Analysis":
    st.markdown('<div class="page-title">🧬 Defense Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Robustness recovery via adversarial training and defense filtering</div>', unsafe_allow_html=True)

    if not selected_attacks:
        st.warning("Select at least one attack to view defenses.")
    else:
        # KPIs
        def_accs, improvements = [], []
        for atk in selected_attacks:
            defr = data[atk].get('defense_results', {})
            entry = next((e for e in defr.get('after_attack_defense', [])
                         if e.get('attack_percentage') == attack_ratio), None)
            if entry:
                def_accs.append(entry.get('defense_accuracy', 0))
                imp = entry.get('improvement',
                    entry.get('defense_accuracy', 0) - data[atk]['baseline_accuracy'])
                improvements.append(imp)

        avg_def = sum(def_accs) / len(def_accs) if def_accs else 0
        avg_imp = sum(improvements) / len(improvements) if improvements else 0

        c1, c2 = st.columns(2)
        with c1:
            kpi("Avg Defense Accuracy", f"{avg_def:.1f}%",
                delta=f"At {attack_ratio}% attack ratio", delta_type="neu", accent="#4ade80")
        with c2:
            kpi("Avg Improvement", f"+{avg_imp:.2f}%",
                delta="vs undefended model", delta_type="pos", accent="#4ade80")

        # ── Before vs After Defense bars ─────────────
        st.markdown(f'<div class="section-header">Attack at {attack_ratio}% vs Defense Accuracy at {attack_ratio}%</div>', unsafe_allow_html=True)

        fig_def = go.Figure()
        attack_vals, def_vals = [], []
        for atk in selected_attacks:
            # "Before defense" = accuracy after attack at selected ratio
            atk_entry = next((r for r in data[atk]['attack_results'] if r['attack_percentage'] == attack_ratio), None)
            a = atk_entry['accuracy_after_attack'] if atk_entry else data[atk]['baseline_accuracy']
            # "After defense" = defense_accuracy at selected ratio
            defr = data[atk].get('defense_results', {})
            def_entry = next((e for e in defr.get('after_attack_defense', [])
                             if e.get('attack_percentage') == attack_ratio), None)
            d = def_entry.get('defense_accuracy', a) if def_entry else a
            attack_vals.append(a)
            def_vals.append(d)

        fig_def.add_trace(go.Bar(
            name=f"After Attack ({attack_ratio}%)", x=selected_attacks, y=attack_vals,
            marker_color=[hex_to_rgba(COLORS.get(a, "#38bdf8"), 0.25) for a in selected_attacks],
            marker_line_color=[COLORS.get(a, "#38bdf8") for a in selected_attacks],
            marker_line_width=1.5,
            text=[f"{v:.2f}%" for v in attack_vals], textposition='auto',
        ))
        fig_def.add_trace(go.Bar(
            name=f"After Defense ({attack_ratio}%)", x=selected_attacks, y=def_vals,
            marker_color=[COLORS.get(a, "#38bdf8") for a in selected_attacks],
            text=[f"{v:.2f}%" for v in def_vals], textposition='auto',
            textfont=dict(color='#0d1220'),
        ))
        fig_def.update_layout(
            **PLOT_LAYOUT,
            barmode='group',
            xaxis=axis_style("Attack Vector"),
            yaxis={**axis_style("Accuracy (%)"), "range": [75, 105]},
            height=380,
        )
        st.plotly_chart(fig_def, use_container_width=True)

        # ── Defense Recovery Trend ───────────────────
        st.markdown('<div class="section-header">Defense Recovery Trend (All Ratios)</div>', unsafe_allow_html=True)

        fig_rec = go.Figure()
        ratios = [1, 2, 3, 4, 5]
        for atk in selected_attacks:
            defr = data[atk].get('defense_results', {})
            y_def, y_base = [], []
            for r in ratios:
                entry = next((e for e in defr.get('after_attack_defense', [])
                              if e.get('attack_percentage') == r), None)
                y_def.append(entry.get('defense_accuracy', None) if entry else None)
                y_base.append(entry.get('baseline_accuracy', None) if entry else None)

            color = COLORS.get(atk, "#38bdf8")
            fig_rec.add_trace(go.Scatter(
                x=ratios, y=y_def, name=f"{atk} Defense",
                mode='lines+markers',
                line=dict(color=color, width=2.5),
                marker=dict(size=7),
            ))
            if any(v is not None for v in y_base):
                fig_rec.add_trace(go.Scatter(
                    x=ratios, y=y_base, name=f"{atk} Undefended",
                    mode='lines',
                    line=dict(color=color, width=1.5, dash='dot'),
                    opacity=0.4, showlegend=False,
                ))

        fig_rec.add_vline(x=attack_ratio, line_dash="dash", line_color="#475569",
                          annotation_text=f"  {attack_ratio}%",
                          annotation_font=dict(color="#64748b", size=11))
        fig_rec.update_layout(
            **PLOT_LAYOUT,
            xaxis={**axis_style("Attack Ratio (%)"), "tickvals": ratios},
            yaxis=axis_style("Accuracy (%)"),
            height=400,
            hovermode="x unified",
        )
        st.plotly_chart(fig_rec, use_container_width=True)