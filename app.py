"""
app.py
Crop Yield Prediction — Smart Agriculture Dashboard
Run with:  streamlit run app.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ── Plotly (optional – graceful fallback) ──────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY = True
except ImportError:
    PLOTLY = False

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌾 Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global */
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

    /* ── Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3c2e 0%, #0d2318 100%);
    }
    [data-testid="stSidebar"] * { color: #e8f5e9 !important; }

    /* ── Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        border-radius: 12px;
        padding: 16px 20px;
        color: white !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    [data-testid="stMetricValue"]  { color: #a5d6a7 !important; font-size: 2rem !important; }
    [data-testid="stMetricLabel"]  { color: #c8e6c9 !important; }

    /* ── Section headers */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2e7d32;
        border-left: 5px solid #66bb6a;
        padding-left: 12px;
        margin: 24px 0 12px;
    }

    /* ── Prediction result box */
    .pred-box {
        background: linear-gradient(135deg, #1b5e20 0%, #388e3c 100%);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        box-shadow: 0 6px 24px rgba(0,0,0,0.3);
        margin-top: 20px;
    }
    .pred-box h2 { color: #ffffff; margin-bottom: 8px; font-size: 1.2rem; }
    .pred-box .yield-val { color: #a5d6a7; font-size: 3.5rem; font-weight: 800; }
    .pred-box .unit { color: #81c784; font-size: 1.1rem; }

    /* ── Tech chip */
    .tech-chip {
        display: inline-block;
        background: #e8f5e9;
        color: #1b5e20;
        border-radius: 20px;
        padding: 4px 14px;
        margin: 4px 3px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #a5d6a7;
    }

    /* ── Info card */
    .info-card {
        background: #f1f8e9;
        border-left: 4px solid #66bb6a;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 14px;
    }
    .info-card h4 { color: #2e7d32; margin: 0 0 6px; }
    .info-card p  { color: #37474f; margin: 0; font-size: 0.92rem; }

    /* ── Footer */
    .footer {
        text-align: center;
        color: #90a4ae;
        font-size: 0.78rem;
        margin-top: 48px;
        padding-top: 16px;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper: load resources (cached) ───────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("dataset/crop_data.csv")

@st.cache_resource
def load_model():
    model   = joblib.load("model/yield_model.pkl")
    le_soil = joblib.load("model/le_soil.pkl")
    le_crop = joblib.load("model/le_crop.pkl")
    return model, le_soil, le_crop


# ── Matplotlib seaborn figure helper ──────────────────────────────
def mpl_fig(figsize=(9, 4)):
    sns.set_theme(style="whitegrid", palette="Set2")
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# ══════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌾 Smart Agriculture")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Data Analysis", "📈 Visualization", "🤖 Yield Prediction"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown("🌲 Random Forest Regressor")
    st.markdown("📈 R² ≈ 96.34%")
    st.markdown("📦 Scikit-learn · Joblib")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("🗂️ 1 000 crop records")
    st.markdown("📌 9 features · 8 crop types")


# ══════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════
if page == "🏠 Home":

    # Hero
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1b5e20,#2e7d32,#388e3c);
                border-radius:20px;padding:44px 40px;margin-bottom:32px;
                box-shadow:0 8px 30px rgba(0,0,0,0.25)'>
      <h1 style='color:#ffffff;font-size:2.4rem;margin:0 0 8px'>
        🌾 Crop Yield Prediction
      </h1>
      <h3 style='color:#a5d6a7;font-weight:400;margin:0 0 18px;font-size:1.2rem'>
        Smart Agriculture Analytics Dashboard
      </h3>
      <p style='color:#c8e6c9;max-width:700px;font-size:0.97rem;line-height:1.7'>
        An end-to-end machine learning system that analyses soil nutrients,
        weather patterns and crop characteristics to <strong style='color:#fff'>
        predict agricultural yield</strong> with ~96% accuracy — helping farmers
        make data-driven decisions and optimise resource usage.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Model Accuracy", "96.34%", "R² Score")
    c2.metric("🌱 Crop Types",     "8",       "Varieties")
    c3.metric("📊 Data Points",    "1,000",   "Records")
    c4.metric("🔬 Features",       "8",       "Input variables")

    st.markdown("---")

    # Two-column info
    left, right = st.columns(2)

    with left:
        st.markdown('<p class="section-title">📚 About This Project</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
          <h4>🎯 Objective</h4>
          <p>Predict crop yield (tonnes/hectare) from soil nutrients, climate data,
          soil type and crop variety using a Random Forest Regressor.</p>
        </div>
        <div class="info-card">
          <h4>🧠 Why Random Forest?</h4>
          <p>Random Forest excels on tabular agricultural data — it handles
          non-linear relationships, mixed feature types, and is robust to
          noise and outliers, making it ideal for farm datasets.</p>
        </div>
        <div class="info-card">
          <h4>📐 Methodology</h4>
          <p>EDA → Feature Engineering → Label Encoding → 80/20 Train-Test Split
          → Random Forest Training → Evaluation (R², MAE, MSE) → Deployment.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-title">🛠️ Technology Stack</p>', unsafe_allow_html=True)
        techs = ["Python 3", "Pandas", "NumPy", "Scikit-learn",
                 "Random Forest", "Matplotlib", "Seaborn",
                 "Streamlit", "Joblib"]
        chips = "".join(f'<span class="tech-chip">{t}</span>' for t in techs)
        st.markdown(chips, unsafe_allow_html=True)

    with right:
        st.markdown('<p class="section-title">🌍 Real-World Applications</p>', unsafe_allow_html=True)
        apps = [
            ("🚜", "Precision Farming",  "Allocate seeds, water and fertiliser exactly where needed."),
            ("📦", "Supply Chain",       "Forecast harvest volumes to optimise logistics and storage."),
            ("📋", "Insurance",          "Estimate expected yield for underwriting crop insurance policies."),
            ("🏛️", "Policy Making",       "Support government food-security planning and subsidy decisions."),
            ("💧", "Water Management",   "Predict irrigation requirements to reduce water waste."),
            ("🌱", "Sustainability",     "Minimise chemical usage through targeted nutrient recommendations."),
        ]
        for icon, title, desc in apps:
            st.markdown(f"""
            <div class="info-card">
              <h4>{icon} {title}</h4>
              <p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown('<p class="footer">Crop Yield Prediction · Built with Streamlit & Scikit-learn · Smart Agriculture ML Project</p>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Data Analysis":

    st.markdown("## 📊 Exploratory Data Analysis")
    df = load_data()

    # ── Dataset preview ──
    st.markdown('<p class="section-title">📋 Dataset Preview</p>', unsafe_allow_html=True)
    st.dataframe(df.head(15), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows",    df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    # ── Statistical summary ──
    st.markdown('<p class="section-title">📐 Statistical Summary</p>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)

    # ── Data types & nulls ──
    st.markdown('<p class="section-title">🔍 Data Types & Null Counts</p>', unsafe_allow_html=True)
    info_df = pd.DataFrame({
        "Data Type":    df.dtypes.astype(str),
        "Null Count":   df.isnull().sum(),
        "Null %":       (df.isnull().mean() * 100).round(2)
    })
    st.dataframe(info_df, use_container_width=True)

    # ── Correlation heatmap ──
    st.markdown('<p class="section-title">🔗 Correlation Heatmap</p>', unsafe_allow_html=True)
    num_df = df.select_dtypes(include=np.number)
    fig, ax = mpl_fig((10, 6))
    mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
    sns.heatmap(num_df.corr(), mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Yield distribution ──
    st.markdown('<p class="section-title">📊 Yield Distribution</p>', unsafe_allow_html=True)
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df['Yield'], bins=30, color='#2e7d32', edgecolor='white', alpha=0.85)
    axes[0].set_title("Yield Histogram", fontweight='bold')
    axes[0].set_xlabel("Yield (tonnes/ha)")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(df['Yield'].mean(), color='#ff5722', linestyle='--', label=f"Mean: {df['Yield'].mean():.1f}")
    axes[0].legend()

    axes[1].boxplot(df['Yield'], vert=False, patch_artist=True,
                    boxprops=dict(facecolor='#a5d6a7', color='#1b5e20'),
                    medianprops=dict(color='#ff5722', linewidth=2))
    axes[1].set_title("Yield Boxplot", fontweight='bold')
    axes[1].set_xlabel("Yield (tonnes/ha)")

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── Crop type distribution ──
    st.markdown('<p class="section-title">🌱 Crop Type Distribution</p>', unsafe_allow_html=True)
    fig3, ax3 = mpl_fig((9, 4))
    counts = df['Crop_Type'].value_counts()
    bars = ax3.bar(counts.index, counts.values,
                   color=plt.cm.Set2(np.linspace(0, 1, len(counts))),
                   edgecolor='white', linewidth=0.8)
    ax3.set_title("Records per Crop Type", fontweight='bold', fontsize=13)
    ax3.set_xlabel("Crop Type")
    ax3.set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 str(val), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — VISUALIZATION
# ══════════════════════════════════════════════════════════════════
elif page == "📈 Visualization":

    st.markdown("## 📈 Agricultural Trend Visualizations")
    df = load_data()

    palette = plt.cm.Set2(np.linspace(0, 1, df['Crop_Type'].nunique()))

    # ── Rainfall vs Yield ──
    st.markdown('<p class="section-title">🌧️ Rainfall vs Crop Yield</p>', unsafe_allow_html=True)
    fig, ax = mpl_fig((10, 4))
    crops   = df['Crop_Type'].unique()
    colors  = {c: plt.cm.tab10(i/len(crops)) for i, c in enumerate(crops)}
    for crop in crops:
        sub = df[df['Crop_Type'] == crop]
        ax.scatter(sub['Rainfall'], sub['Yield'], label=crop, alpha=0.55,
                   s=25, color=colors[crop])
    ax.set_xlabel("Rainfall (mm)")
    ax.set_ylabel("Yield (tonnes/ha)")
    ax.set_title("Rainfall vs Yield by Crop Type", fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Temperature vs Yield ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">🌡️ Temperature vs Yield</p>', unsafe_allow_html=True)
        fig, ax = mpl_fig((6, 4))
        ax.scatter(df['Temperature'], df['Yield'], alpha=0.4, c=df['Yield'],
                   cmap='RdYlGn', s=20, edgecolors='none')
        # trend line
        z = np.polyfit(df['Temperature'], df['Yield'], 2)
        p = np.poly1d(z)
        xs = np.linspace(df['Temperature'].min(), df['Temperature'].max(), 200)
        ax.plot(xs, p(xs), color='#d32f2f', linewidth=2, label='Trend')
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Yield (tonnes/ha)")
        ax.set_title("Temperature vs Yield", fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<p class="section-title">💧 Humidity vs Yield</p>', unsafe_allow_html=True)
        fig, ax = mpl_fig((6, 4))
        ax.scatter(df['Humidity'], df['Yield'], alpha=0.4, c=df['Yield'],
                   cmap='Blues', s=20, edgecolors='none')
        z = np.polyfit(df['Humidity'], df['Yield'], 1)
        p = np.poly1d(z)
        xs = np.linspace(df['Humidity'].min(), df['Humidity'].max(), 200)
        ax.plot(xs, p(xs), color='#1565c0', linewidth=2, label='Trend')
        ax.set_xlabel("Humidity (%)")
        ax.set_ylabel("Yield (tonnes/ha)")
        ax.set_title("Humidity vs Yield", fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Crop Type vs Average Yield ──
    st.markdown('<p class="section-title">🌾 Crop Type vs Average Yield</p>', unsafe_allow_html=True)
    avg_yield = df.groupby('Crop_Type')['Yield'].agg(['mean', 'std']).reset_index()
    avg_yield.columns = ['Crop_Type', 'Mean_Yield', 'Std_Yield']
    avg_yield = avg_yield.sort_values('Mean_Yield', ascending=False)

    fig, ax = mpl_fig((10, 4))
    bar_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(avg_yield)))
    bars = ax.bar(avg_yield['Crop_Type'], avg_yield['Mean_Yield'],
                  yerr=avg_yield['Std_Yield'], color=bar_colors,
                  edgecolor='white', linewidth=0.8,
                  error_kw=dict(ecolor='#455a64', capsize=5, linewidth=1.2))
    ax.set_title("Average Yield per Crop Type (with Std Dev)", fontweight='bold', fontsize=13)
    ax.set_xlabel("Crop Type")
    ax.set_ylabel("Average Yield (tonnes/ha)")
    for bar, val in zip(bars, avg_yield['Mean_Yield']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Nutrient box plots ──
    st.markdown('<p class="section-title">🧪 Nutrient Distribution by Crop</p>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, nutrient in zip(axes, ['Nitrogen', 'Phosphorus', 'Potassium']):
        data_by_crop = [df[df['Crop_Type'] == c][nutrient].values for c in sorted(df['Crop_Type'].unique())]
        bp = ax.boxplot(data_by_crop, patch_artist=True,
                        boxprops=dict(facecolor='#c8e6c9', color='#2e7d32'),
                        medianprops=dict(color='#e53935', linewidth=2),
                        whiskerprops=dict(color='#2e7d32'),
                        capprops=dict(color='#2e7d32'))
        ax.set_xticklabels(sorted(df['Crop_Type'].unique()), rotation=35, ha='right', fontsize=8)
        ax.set_title(f"{nutrient} by Crop Type", fontweight='bold')
        ax.set_ylabel(f"{nutrient} level")
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.suptitle("Nutrient Levels across Crop Types", fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Soil type yield ──
    st.markdown('<p class="section-title">🪨 Soil Type vs Average Yield</p>', unsafe_allow_html=True)
    soil_yield = df.groupby('Soil_Type')['Yield'].mean().sort_values(ascending=True)
    fig, ax = mpl_fig((8, 4))
    bars = ax.barh(soil_yield.index, soil_yield.values,
                   color=plt.cm.YlOrBr(np.linspace(0.4, 0.9, len(soil_yield))),
                   edgecolor='white')
    for bar, val in zip(bars, soil_yield.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va='center', fontsize=10, fontweight='bold')
    ax.set_xlabel("Average Yield (tonnes/ha)")
    ax.set_title("Average Yield by Soil Type", fontweight='bold', fontsize=13)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# PAGE 4 — YIELD PREDICTION
# ══════════════════════════════════════════════════════════════════
elif page == "🤖 Yield Prediction":

    st.markdown("## 🤖 Crop Yield Prediction")
    st.markdown("Enter the agricultural parameters below to predict the expected crop yield.")

    model, le_soil, le_crop = load_model()
    df = load_data()

    # ── Model performance banner ──
    st.markdown("""
    <div style='background:linear-gradient(90deg,#1b5e20,#388e3c);
                border-radius:12px;padding:18px 28px;margin-bottom:24px;
                display:flex;gap:32px;flex-wrap:wrap'>
      <div style='color:white'><b style='color:#a5d6a7;font-size:1.3rem'>96.34%</b><br><small>R² Score</small></div>
      <div style='color:white'><b style='color:#a5d6a7;font-size:1.3rem'>3.01</b><br><small>MAE (t/ha)</small></div>
      <div style='color:white'><b style='color:#a5d6a7;font-size:1.3rem'>14.44</b><br><small>MSE</small></div>
      <div style='color:white'><b style='color:#a5d6a7;font-size:1.3rem'>3.80</b><br><small>RMSE</small></div>
      <div style='color:white'><b style='color:#a5d6a7;font-size:1.3rem'>150</b><br><small>Trees</small></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input form ──
    st.markdown('<p class="section-title">🌱 Enter Crop Parameters</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🧪 Soil Nutrients**")
        nitrogen   = st.slider("Nitrogen (N)",    20.0, 120.0, 70.0, 1.0,
                                help="Nitrogen content in soil (kg/ha)")
        phosphorus = st.slider("Phosphorus (P)",  10.0,  80.0, 45.0, 1.0,
                                help="Phosphorus content in soil (kg/ha)")
        potassium  = st.slider("Potassium (K)",   15.0,  90.0, 50.0, 1.0,
                                help="Potassium content in soil (kg/ha)")

    with col2:
        st.markdown("**🌤️ Climate Conditions**")
        temperature = st.slider("Temperature (°C)", 15.0, 40.0, 27.0, 0.5,
                                 help="Average growing season temperature")
        humidity    = st.slider("Humidity (%)",     30.0, 95.0, 65.0, 1.0,
                                 help="Average relative humidity")
        rainfall    = st.slider("Rainfall (mm)",   200.0, 1500.0, 850.0, 10.0,
                                 help="Annual rainfall")

    with col3:
        st.markdown("**🌾 Crop Details**")
        crop_type = st.selectbox("Crop Type", sorted(df['Crop_Type'].unique()),
                                  help="Select the crop variety")
        soil_type = st.selectbox("Soil Type", sorted(df['Soil_Type'].unique()),
                                  help="Select the soil type")

        st.markdown("**📊 Input Summary**")
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Nitrogen | {nitrogen} |
        | Phosphorus | {phosphorus} |
        | Potassium | {potassium} |
        | Temperature | {temperature}°C |
        | Humidity | {humidity}% |
        | Rainfall | {rainfall} mm |
        | Crop | {crop_type} |
        | Soil | {soil_type} |
        """)

    st.markdown("---")
    _, btn_col, _ = st.columns([2, 1, 2])
    predict_clicked = btn_col.button("🔮  Predict Yield", type="primary", use_container_width=True)

    if predict_clicked:
        soil_enc = le_soil.transform([soil_type])[0]
        crop_enc = le_crop.transform([crop_type])[0]

        input_data = pd.DataFrame([[nitrogen, phosphorus, potassium,
                                     temperature, humidity, rainfall,
                                     soil_enc, crop_enc]],
                                   columns=['Nitrogen', 'Phosphorus', 'Potassium',
                                            'Temperature', 'Humidity', 'Rainfall',
                                            'Soil_Type_Enc', 'Crop_Type_Enc'])

        prediction = model.predict(input_data)[0]
        prediction = max(0.0, prediction)

        # ── Result display ──
        res_col, gauge_col = st.columns([1, 1])

        with res_col:
            st.markdown(f"""
            <div class="pred-box">
              <h2>🌾 Predicted Crop Yield</h2>
              <div class="yield-val">{prediction:.2f}</div>
              <div class="unit">tonnes per hectare</div>
              <br>
              <div style="color:#c8e6c9;font-size:0.9rem">
                Crop: <b>{crop_type}</b> &nbsp;|&nbsp; Soil: <b>{soil_type}</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Rating
            avg_crop_yield = df[df['Crop_Type'] == crop_type]['Yield'].mean()
            if prediction >= avg_crop_yield * 1.15:
                rating, color, icon = "Excellent", "#2e7d32", "🌟"
            elif prediction >= avg_crop_yield * 0.90:
                rating, color, icon = "Good",      "#1565c0", "✅"
            elif prediction >= avg_crop_yield * 0.70:
                rating, color, icon = "Average",   "#e65100", "⚠️"
            else:
                rating, color, icon = "Below Average", "#c62828", "❌"

            st.markdown(f"""
            <div style='background:{color}22;border:2px solid {color};
                        border-radius:10px;padding:14px 20px;margin-top:16px;text-align:center'>
              <b style='color:{color};font-size:1.1rem'>{icon} Yield Rating: {rating}</b><br>
              <span style='color:#546e7a;font-size:0.85rem'>
                Crop average: {avg_crop_yield:.2f} t/ha &nbsp;|&nbsp; Your prediction: {prediction:.2f} t/ha
              </span>
            </div>
            """, unsafe_allow_html=True)

        with gauge_col:
            st.markdown("**📊 Feature Impact Snapshot**")
            # Simple bar chart of normalised inputs
            inputs_norm = {
                'Nitrogen':   (nitrogen   - 20)  / (120 - 20),
                'Phosphorus': (phosphorus - 10)  / (80  - 10),
                'Potassium':  (potassium  - 15)  / (90  - 15),
                'Temperature':(temperature- 15)  / (40  - 15),
                'Humidity':   (humidity   - 30)  / (95  - 30),
                'Rainfall':   (rainfall   - 200) / (1500- 200),
            }
            fig, ax = mpl_fig((6, 4))
            names  = list(inputs_norm.keys())
            values = list(inputs_norm.values())
            colors = ['#2e7d32' if v >= 0.5 else '#e65100' for v in values]
            ax.barh(names, values, color=colors, edgecolor='white')
            ax.set_xlim(0, 1)
            ax.axvline(0.5, color='#90a4ae', linestyle='--', linewidth=1)
            ax.set_xlabel("Normalised Value (0 = min, 1 = max)")
            ax.set_title("Input Parameter Levels", fontweight='bold')
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Feature importance
            st.markdown("**🌲 Model Feature Importance**")
            if os.path.exists("model/feature_importance.png"):
                st.image("model/feature_importance.png")

        # Recommendations
        st.markdown('<p class="section-title">💡 Agronomic Recommendations</p>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)

        optimal_temp = 27.0
        with r1:
            n_status = "✅ Optimal" if 60 <= nitrogen <= 90 else ("⬆️ Increase" if nitrogen < 60 else "⬇️ Reduce")
            st.info(f"**Nitrogen:** {n_status}\n\nCurrent: {nitrogen} kg/ha\nOptimal range: 60–90 kg/ha")
        with r2:
            t_status = "✅ Optimal" if abs(temperature - optimal_temp) <= 3 else ("🌡️ Too Hot" if temperature > optimal_temp + 3 else "❄️ Too Cold")
            st.info(f"**Temperature:** {t_status}\n\nCurrent: {temperature}°C\nOptimal: ~{optimal_temp}°C")
        with r3:
            r_status = "✅ Adequate" if 600 <= rainfall <= 1200 else ("⬆️ Irrigate" if rainfall < 600 else "⬇️ Drainage needed")
            st.info(f"**Rainfall:** {r_status}\n\nCurrent: {rainfall} mm\nOptimal: 600–1200 mm")

    st.markdown('<p class="footer">Crop Yield Prediction · Smart Agriculture Dashboard</p>',
                unsafe_allow_html=True)
