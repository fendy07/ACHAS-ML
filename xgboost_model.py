import joblib
import pandas as pd
from xgboost import XGBRegressor

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    "data/physiological_timeseries.csv"
)

# =========================================================
# FEATURES
# =========================================================

features = [
    "heart_rate",
    "skin_temp",
    "env_temp",
    "humidity",
    "wind_speed",
    "activity"
]

target = "core_temp"

X = df[features]
y = df[target]

# =========================================================
# MODEL
# =========================================================

model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    objective="reg:squarederror"
)

model.fit(X, y)

# =========================================================
# SAVE
# =========================================================

joblib.dump(
    model,
    "model/xgb_hybrid_model.pkl"
)

print("Saved!")