"""
train_model.py
Crop Yield Prediction — Model Training Script
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
print("=" * 55)
print("  CROP YIELD PREDICTION — MODEL TRAINING")
print("=" * 55)

df = pd.read_csv("dataset/crop_data.csv")
print(f"\n✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(df.head(3).to_string())

# ─────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────
print("\n📋 Missing values:")
print(df.isnull().sum())

# Encode categoricals
le_soil = LabelEncoder()
le_crop = LabelEncoder()
df['Soil_Type_Enc']  = le_soil.fit_transform(df['Soil_Type'])
df['Crop_Type_Enc']  = le_crop.fit_transform(df['Crop_Type'])

# Save encoders
os.makedirs("model", exist_ok=True)
joblib.dump(le_soil, "model/le_soil.pkl")
joblib.dump(le_crop, "model/le_crop.pkl")

# ─────────────────────────────────────────
# 3. FEATURE / TARGET SPLIT
# ─────────────────────────────────────────
FEATURES = ['Nitrogen', 'Phosphorus', 'Potassium',
            'Temperature', 'Humidity', 'Rainfall',
            'Soil_Type_Enc', 'Crop_Type_Enc']
TARGET   = 'Yield'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

print(f"\n🔀 Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────
# 4. TRAIN RANDOM FOREST
# ─────────────────────────────────────────
print("\n🌲 Training Random Forest Regressor …")
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✅ Training complete.")

# ─────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────
y_pred = model.predict(X_test)

r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n" + "─" * 40)
print("  📊 MODEL EVALUATION RESULTS")
print("─" * 40)
print(f"  R²  Score  : {r2:.4f}  ({r2*100:.2f}%)")
print(f"  MAE        : {mae:.4f}")
print(f"  MSE        : {mse:.4f}")
print(f"  RMSE       : {rmse:.4f}")
print("─" * 40)

# ─────────────────────────────────────────
# 6. FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────
feat_imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feat_imp)))
feat_imp.plot(kind='bar', ax=ax, color=colors, edgecolor='white', linewidth=0.6)
ax.set_title("Feature Importance — Random Forest", fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel("Feature", fontsize=11)
ax.set_ylabel("Importance Score", fontsize=11)
ax.tick_params(axis='x', rotation=35)
ax.grid(axis='y', alpha=0.3, linestyle='--')
for bar, val in zip(ax.patches, feat_imp.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.3f}", ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig("model/feature_importance.png", dpi=150)
plt.close()
print("\n📊 Feature importance plot saved → model/feature_importance.png")

# ─────────────────────────────────────────
# 7. SAVE MODEL
# ─────────────────────────────────────────
joblib.dump(model, "model/yield_model.pkl")
print("💾 Model saved → model/yield_model.pkl")
print("\n🎉 All done! Run  streamlit run app.py  to launch the dashboard.\n")
