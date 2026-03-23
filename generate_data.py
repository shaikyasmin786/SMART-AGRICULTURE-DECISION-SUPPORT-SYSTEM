import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 1000

soil_types = ['Sandy', 'Loamy', 'Clay', 'Silty', 'Peaty']
crop_types = ['Wheat', 'Rice', 'Maize', 'Soybean', 'Cotton', 'Sugarcane', 'Barley', 'Millet']

crop_base_yield = {
    'Wheat': 35, 'Rice': 45, 'Maize': 50, 'Soybean': 28,
    'Cotton': 20, 'Sugarcane': 80, 'Barley': 30, 'Millet': 22
}

crop_type = np.random.choice(crop_types, n)
soil_type = np.random.choice(soil_types, n)

nitrogen    = np.random.uniform(20, 120, n)
phosphorus  = np.random.uniform(10, 80, n)
potassium   = np.random.uniform(15, 90, n)
temperature = np.random.uniform(15, 40, n)
humidity    = np.random.uniform(30, 95, n)
rainfall    = np.random.uniform(200, 1500, n)

base = np.array([crop_base_yield[c] for c in crop_type])
yield_val = (
    base
    + 0.10 * nitrogen
    + 0.08 * phosphorus
    + 0.06 * potassium
    + -0.30 * abs(temperature - 27)
    + 0.04 * humidity
    + 0.015 * rainfall
    + np.random.normal(0, 3, n)
)
yield_val = np.clip(yield_val, 5, 120)

os.makedirs('dataset', exist_ok=True)

df = pd.DataFrame({
    'Nitrogen':    nitrogen.round(2),
    'Phosphorus':  phosphorus.round(2),
    'Potassium':   potassium.round(2),
    'Temperature': temperature.round(2),
    'Humidity':    humidity.round(2),
    'Rainfall':    rainfall.round(2),
    'Soil_Type':   soil_type,
    'Crop_Type':   crop_type,
    'Yield':       yield_val.round(2)
})

df.to_csv('dataset/crop_data.csv', index=False)
print('✅ Dataset created successfully!')
print(f'   Shape: {df.shape}')
print(df.head(3))