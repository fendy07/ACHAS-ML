import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

np.random.seed(42)

TOTAL_MINUTES = 24 * 60   # 1 day
dt = 1                    # 1-minute interval

timestamps = pd.date_range(
    start="2026-01-01",
    periods=TOTAL_MINUTES,
    freq="1min"
)

# =========================================================
# STATIC PROFILE
# =========================================================

age = 30
height = 1.72
weight = 70
bmi = weight / (height ** 2)

# =========================================================
# STORAGE
# =========================================================

heart_rate = []
skin_temp = []
env_temp = []
humidity = []
wind_speed = []
activity = []
core_temp = []

# =========================================================
# INITIAL CONDITIONS
# =========================================================

t_core = 37.0
t_skin = 34.5
hr = 72

# =========================================================
# SIMULATION LOOP
# =========================================================

for t in range(TOTAL_MINUTES):

    hour = t / 60

    # -----------------------------------------------------
    # 1. Circadian rhythm
    # -----------------------------------------------------

    circadian = 0.3 * np.sin(2 * np.pi * (hour - 4) / 24)

    # -----------------------------------------------------
    # 2. Environment dynamics
    # -----------------------------------------------------

    env = (
        28
        + 4 * np.sin(2 * np.pi * (hour - 6) / 24)
        + np.random.normal(0, 0.3)
    )

    hum = (
        65
        - 10 * np.sin(2 * np.pi * (hour - 6) / 24)
        + np.random.normal(0, 2)
    )

    wind = max(
        0,
        np.random.normal(1.5, 0.5)
    )

    # -----------------------------------------------------
    # 3. Activity pattern
    # -----------------------------------------------------

    # 0=sit, 1=walk, 2=exercise

    if 7 <= hour <= 8:
        act = 2
    elif 18 <= hour <= 19:
        act = 1
    else:
        act = 0

    # random spikes
    if np.random.rand() < 0.01:
        act = 2

    # -----------------------------------------------------
    # 4. Heart rate dynamics
    # -----------------------------------------------------

    target_hr = {
        0: 72,
        1: 95,
        2: 130
    }[act]

    # smooth HR transition
    hr += 0.1 * (target_hr - hr)

    hr_noise = np.random.normal(0, 2)

    hr_measured = hr + hr_noise

    # -----------------------------------------------------
    # 5. Core temperature dynamics
    # -----------------------------------------------------

    metabolic_heat = 0.002 * (hr - 70)

    environmental_heat = (
        0.0015 * (env - 25)
        + 0.0005 * (hum - 50)
    )

    cooling = (
        0.001 * wind * (t_skin - env)
    )

    # thermal inertia
    dT = (
        circadian
        + metabolic_heat
        + environmental_heat
        - cooling
    )

    t_core += 0.02 * dT

    # physiological bounds
    t_core = np.clip(t_core, 36.0, 39.5)

    # -----------------------------------------------------
    # 6. Skin temperature dynamics
    # -----------------------------------------------------

    t_skin += (
        0.05 * (t_core - t_skin)
        + 0.01 * (env - t_skin)
    )

    t_skin_measured = (
        t_skin
        + np.random.normal(0, 0.2)
    )

    # -----------------------------------------------------
    # STORE
    # -----------------------------------------------------

    heart_rate.append(hr_measured)
    skin_temp.append(t_skin_measured)
    env_temp.append(env)
    humidity.append(hum)
    wind_speed.append(wind)
    activity.append(act)
    core_temp.append(t_core)

# =========================================================
# DATAFRAME
# =========================================================

df = pd.DataFrame({
    "timestamp": timestamps,
    "age": age,
    "height": height,
    "weight": weight,
    "bmi": bmi,
    "heart_rate": heart_rate,
    "skin_temp": skin_temp,
    "env_temp": env_temp,
    "humidity": humidity,
    "wind_speed": wind_speed,
    "activity": activity,
    "core_temp": core_temp
})

# =========================================================
# SAVE
# =========================================================

df.to_csv("data/physiological_timeseries.csv", index=False)

print(df.head())

print("\nSaved:")
print("data/physiological_timeseries.csv")