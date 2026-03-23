# 🌾 Crop Yield Prediction — Smart Agriculture ML Project

A professional end-to-end machine learning web application that predicts crop yield using soil nutrients, climate data and crop characteristics.

---

## 📁 Project Structure

```
Crop-Yield-Prediction/
│
├── dataset/
│     └── crop_data.csv          ← 1000-row agricultural dataset
│
├── model/
│     ├── yield_model.pkl        ← Trained Random Forest model
│     ├── le_soil.pkl            ← Soil type label encoder
│     ├── le_crop.pkl            ← Crop type label encoder
│     └── feature_importance.png ← Feature importance chart
│
├── analysis.ipynb               ← Full EDA + ML notebook
├── train_model.py               ← Standalone training script
├── app.py                       ← Streamlit web dashboard
├── generate_data.py             ← Dataset generation script
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset (already done)
python generate_data.py

# 3. Train the model (already done)
python train_model.py

# 4. Launch the dashboard
streamlit run app.py
```

---

## 🧠 Technology Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3 |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| ML | Scikit-learn (Random Forest) |
| Model Saving | Joblib |
| Web App | Streamlit |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| R² Score | 96.34% |
| MAE | 3.01 t/ha |
| MSE | 14.44 |
| RMSE | 3.80 t/ha |

---

## 🌐 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview, tech stack, real-world applications |
| 📊 Data Analysis | Dataset preview, statistics, correlation heatmap |
| 📈 Visualization | Rainfall/Temperature/Humidity trends, crop comparisons |
| 🤖 Yield Prediction | Interactive form → instant ML prediction |

---

## 📌 Input Features

- **Nitrogen** — Soil nitrogen content (kg/ha)
- **Phosphorus** — Soil phosphorus content (kg/ha)
- **Potassium** — Soil potassium content (kg/ha)
- **Temperature** — Average growing season temperature (°C)
- **Humidity** — Relative humidity (%)
- **Rainfall** — Annual rainfall (mm)
- **Soil_Type** — Sandy / Loamy / Clay / Silty / Peaty
- **Crop_Type** — Wheat / Rice / Maize / Soybean / Cotton / Sugarcane / Barley / Millet

**Target:** `Yield` (tonnes per hectare)

---

## 🌍 Real-World Applications

- Precision farming & resource optimization
- Crop insurance underwriting
- Government food-security planning
- Supply chain & harvest forecasting
- Irrigation and water management
