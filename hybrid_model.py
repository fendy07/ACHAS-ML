import numpy as np
import joblib

# =========================================================
# LOAD XGBOOST MODEL
# =========================================================

xgb_model = joblib.load(
    "model/xgb_hybrid_model.pkl"
    
)

# =========================================================
# ADAPTIVE KALMAN FILTER
# =========================================================

class AdaptiveKalmanFilter:

    def __init__(self):

        # initial state
        self.x = 37.0

        # uncertainty
        self.p = 1.0

        # baseline noise
        self.base_q = 0.01
        self.base_r = 0.1

    # =====================================================
    # ADAPTIVE NOISE ESTIMATION
    # =====================================================

    def estimate_noise(
        self,
        activity,
        wind_speed,
        hr_variability,
        skin_env_delta
    ):

        # -------------------------------------------------
        # PROCESS NOISE (Q)
        # more activity = faster physiological changes
        # -------------------------------------------------

        q = (
            self.base_q
            + 0.01 * activity
            + 0.002 * hr_variability
        )

        # -------------------------------------------------
        # MEASUREMENT NOISE (R)
        # more wind + skin/env mismatch = noisier skin temp
        # -------------------------------------------------

        r = (
            self.base_r
            + 0.02 * wind_speed
            + 0.01 * abs(skin_env_delta)
        )

        return q, r

    # =====================================================
    # UPDATE
    # =====================================================

    def update(
        self,
        measurement,
        activity,
        wind_speed,
        hr_variability,
        skin_env_delta
    ):

        # -------------------------------------------------
        # adaptive noise
        # -------------------------------------------------

        q, r = self.estimate_noise(
            activity,
            wind_speed,
            hr_variability,
            skin_env_delta
        )

        # -------------------------------------------------
        # prediction step
        # -------------------------------------------------

        self.p = self.p + q

        # -------------------------------------------------
        # kalman gain
        # -------------------------------------------------

        k = self.p / (self.p + r)

        # -------------------------------------------------
        # correction
        # -------------------------------------------------

        self.x = self.x + k * (
            measurement - self.x
        )

        # -------------------------------------------------
        # uncertainty update
        # -------------------------------------------------

        self.p = (1 - k) * self.p

        return self.x, k, q, r

# =========================================================
# HYBRID MODEL
# =========================================================

class AdaptiveHybridModel:

    def __init__(self):

        self.filter = AdaptiveKalmanFilter()

        self.prev_hr = 70

    # =====================================================
    # PREDICT
    # =====================================================

    def predict(
        self,
        heart_rate,
        skin_temp,
        env_temp,
        humidity,
        wind_speed,
        activity
    ):

        # -------------------------------------------------
        # XGBOOST INPUT
        # -------------------------------------------------

        X = np.array([[
            heart_rate,
            skin_temp,
            env_temp,
            humidity,
            wind_speed,
            activity
        ]])

        # -------------------------------------------------
        # RAW ML PREDICTION
        # -------------------------------------------------

        raw_prediction = (
            xgb_model.predict(X)[0]
        )

        # -------------------------------------------------
        # DYNAMIC FEATURES
        # -------------------------------------------------

        hr_variability = abs(
            heart_rate - self.prev_hr
        )

        skin_env_delta = (
            skin_temp - env_temp
        )

        self.prev_hr = heart_rate

        # -------------------------------------------------
        # ADAPTIVE KALMAN
        # -------------------------------------------------

        smooth, k, q, r = (
            self.filter.update(
                measurement=raw_prediction,
                activity=activity,
                wind_speed=wind_speed,
                hr_variability=hr_variability,
                skin_env_delta=skin_env_delta
            )
        )

        # -------------------------------------------------
        # OUTPUT
        # -------------------------------------------------

        return {

            "raw_prediction":
                round(float(raw_prediction), 3),

            "smoothed_prediction":
                round(float(smooth), 3),

            "kalman_gain":
                round(float(k), 3),

            "process_noise_q":
                round(float(q), 4),

            "measurement_noise_r":
                round(float(r), 4)
        }