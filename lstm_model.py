# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv("data/physiological_timeseries.csv")

# =========================================================
# 2. FEATURES
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

# =========================================================
# 3. NORMALIZATION
# =========================================================

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = feature_scaler.fit_transform(df[features])

y_scaled = target_scaler.fit_transform(
    df[[target]]
)

# =========================================================
# 4. CREATE SEQUENCES
# =========================================================

SEQ_LEN = 30  # 30 minutes history

X = []
y = []

for i in range(SEQ_LEN, len(df)):

    X.append(
        X_scaled[i-SEQ_LEN:i]
    )

    y.append(
        y_scaled[i]
    )

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================================================
# 5. TRAIN TEST SPLIT
# =========================================================

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# =========================================================
# 6. BUILD MODEL
# =========================================================

model = Sequential([

    LSTM(
        64,
        return_sequences=True,
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    Dropout(0.2),

    LSTM(32),

    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# =========================================================
# 7. TRAINING
# =========================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)

# =========================================================
# 8. PREDICTION
# =========================================================

pred_scaled = model.predict(X_test)

pred = target_scaler.inverse_transform(pred_scaled)
y_true = target_scaler.inverse_transform(y_test)

# =========================================================
# 9. EVALUATION
# =========================================================

mae = mean_absolute_error(y_true, pred)
rmse = np.sqrt(mean_squared_error(y_true, pred))
r2 = r2_score(y_true, pred)

print("\n===== PERFORMANCE =====")
print(f"MAE  : {mae:.3f} °C")
print(f"RMSE : {rmse:.3f} °C")
print(f"R2-Squared   : {r2:.3f}")

# =========================================================
# 10. SAVE MODEL
# =========================================================

model.save("model/lstm_core_temp_model.keras")

print("\nModel saved:")
print("model/lstm_core_temp_model.keras")

# =========================================================
# 11. VISUALIZATION
# =========================================================

plt.figure(figsize=(14, 6))

plt.plot(
    y_true[:300],
    label="True Core Temp"
)

plt.plot(
    pred[:300],
    label="Predicted Core Temp"
)

plt.legend()
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Core Temperature Prediction")
plt.savefig("images/test_prediction_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# =========================================================
# 12. PREDICTION WITH NEW (UNSEEN) DATA
# =========================================================
# Simulate new sensor readings that the model has never seen.
# In production, replace this block with real incoming data.

print("\n===== NEW DATA PREDICTION =====")

# --- 12a. Generate synthetic new data (60 minutes into the future) ---
np.random.seed(42)
n_new = 60  # 60 new minutes

last_row = df.iloc[-1]

new_data = pd.DataFrame({
    "heart_rate":  np.random.normal(last_row["heart_rate"],  2, n_new).clip(50, 180),
    "skin_temp":   np.random.normal(last_row["skin_temp"],   0.3, n_new).clip(30, 40),
    "env_temp":    np.random.normal(last_row["env_temp"],    0.5, n_new).clip(15, 45),
    "humidity":    np.random.normal(last_row["humidity"],     2, n_new).clip(20, 100),
    "wind_speed":  np.random.normal(last_row["wind_speed"],  0.3, n_new).clip(0, 10),
    "activity":    np.random.choice([0, 1, 2], size=n_new, p=[0.7, 0.2, 0.1])
})

print(f"New data shape: {new_data.shape}")
print(new_data.head())

# --- 12b. Combine tail of training data with new data for sequence context ---
# We need at least SEQ_LEN rows of history before we can predict
history_tail = df[features].iloc[-SEQ_LEN:]
combined = pd.concat([history_tail, new_data], ignore_index=True)

# Normalize using the SAME fitted scaler (important!)
combined_scaled = feature_scaler.transform(combined)

# --- 12c. Create sequences from the combined data ---
X_new = []
for i in range(SEQ_LEN, len(combined_scaled)):
    X_new.append(combined_scaled[i - SEQ_LEN : i])

X_new = np.array(X_new)
print(f"X_new shape: {X_new.shape}")

# --- 12d. Predict core temperature ---
pred_new_scaled = model.predict(X_new)
pred_new = target_scaler.inverse_transform(pred_new_scaled)

# --- 12e. Print predictions ---
print(f"\nPredicted core temperatures for next {n_new} minutes:")
print(f"  Min  : {pred_new.min():.3f} °C")
print(f"  Max  : {pred_new.max():.3f} °C")
print(f"  Mean : {pred_new.mean():.3f} °C")
print(f"  Std  : {pred_new.std():.3f} °C")

# --- 12f. Visualize new predictions alongside recent actual data ---
recent_actual = target_scaler.inverse_transform(
    y_scaled[-60:]
)

plt.figure(figsize=(14, 6))

plt.plot(
    range(0, len(recent_actual)),
    recent_actual,
    label="Recent Actual Core Temp",
    color="#2196F3",
    linewidth=2
)

plt.plot(
    range(len(recent_actual), len(recent_actual) + len(pred_new)),
    pred_new,
    label="Forecasted Core Temp (New Data)",
    color="#FF5722",
    linewidth=2,
    linestyle="--"
)

plt.axvline(
    x=len(recent_actual),
    color="gray",
    linestyle=":",
    label="Forecast Start"
)

plt.legend()
plt.xlabel("Time (minutes)")
plt.ylabel("Temperature (°C)")
plt.title("Core Temperature: Actual vs. New Data Forecast")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/new_data_forecast_plot.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nForecast plot saved: images/new_data_forecast_plot.png")