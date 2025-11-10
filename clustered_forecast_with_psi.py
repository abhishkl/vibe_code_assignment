# ============================================================
# CLUSTERED FORECAST WITH MODEL SWITCHING (PSI + EXTERNAL SIGNALS)
# ============================================================
# Generates synthetic car rental demand data for multiple car-class clusters
# (Economy, SUV, Luxury) using external signals (Hotel ADR, Airline ADR, Airline
# Cancellations). Trains multiple model families and uses Population Stability
# Index (PSI) to detect data drift and trigger model switching.
# Forecasts 90 days ahead and logs model changes.#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error


warnings.filterwarnings("ignore")

np.random.seed(42)


# -----------------------------------
# 1. Synthetic External Signals
# -----------------------------------

def generate_external_signals(n_days=270):
    """
    Create synthetic external signals for Hotel ADR, Airline ADR, and Airline Cancellations.
    """
    days = np.arange(n_days)
    hotel_adr = 100 + 10 * np.sin(days / 30) + 0.1 * days + np.random.normal(0, 2, n_days)
    airline_adr = 120 + 15 * np.sin(days / 25) + 0.15 * days + np.random.normal(0, 3, n_days)
    airline_cxl = np.clip(np.random.normal(0.05, 0.02, n_days) + np.sin(days / 50) * 0.02, 0, 0.15)
    return pd.DataFrame({
        'day': days,
        'hotel_adr': hotel_adr,
        'airline_adr': airline_adr,
        'airline_cxl_rate': airline_cxl
    })


external_df = generate_external_signals(270)


# -----------------------------------
# 2. Generate Cluster Data
# -----------------------------------

def generate_cluster_data(cluster_name, base_demand, decay, noise, sensitivity, n_days=270):
    """
    Create synthetic booking demand influenced by external signals.
    Each cluster reacts differently to ADRs and cancellations.
    """
    df = external_df.copy()
    df['cluster'] = cluster_name
    df['base'] = base_demand * np.exp(-decay * df['day'])

    df['bookings'] = (
            df['base']
            + sensitivity[0] * df['hotel_adr']
            - sensitivity[1] * df['airline_adr']
            - sensitivity[2] * df['airline_cxl_rate'] * 1000
            + np.random.normal(0, noise, len(df))
    )
    df['bookings'] = np.maximum(df['bookings'], 0)
    return df[['day', 'hotel_adr', 'airline_adr', 'airline_cxl_rate', 'cluster', 'bookings']]


# Cluster sensitivities: [hotel_sensitivity, airline_sensitivity, cancellation_sensitivity]
clusters = {
    "Economy": generate_cluster_data("Economy", 550, 0.03, 10, [1.0, 0.3, 0.5]),
    "SUV": generate_cluster_data("SUV", 450, 0.05, 15, [0.8, 0.4, 0.8]),
    "Luxury": generate_cluster_data("Luxury", 320, 0.02, 25, [1.5, 0.6, 1.2])
}

df = pd.concat(clusters.values(), ignore_index=True)


# -----------------------------------
# 3. Model Families
# -----------------------------------

def train_linear_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_rf_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


model_families = {"linear": train_linear_model, "rf": train_rf_model}


# -----------------------------------
# 4. PSI Drift Detection
# -----------------------------------

def calculate_psi(expected, actual, buckets=10, eps=1e-6):
    """
    Calculate Population Stability Index between two numeric distributions.
    PSI < 0.1 : No significant drift
    PSI 0.1–0.25 : Moderate drift
    PSI > 0.25 : Significant drift
    """
    expected, actual = np.array(expected), np.array(actual)
    breakpoints = np.percentile(expected, np.arange(0, 100, 100 / buckets))

    expected_bins = np.histogram(expected, bins=breakpoints)[0]
    actual_bins = np.histogram(actual, bins=breakpoints)[0]

    expected_percents = expected_bins / len(expected)
    actual_percents = actual_bins / len(actual)

    psi_values = (expected_percents - actual_percents) * np.log((expected_percents + eps) / (actual_percents + eps))
    psi_total = np.sum(psi_values)
    return psi_total


def drift_detected_psi(train_data, new_data, threshold=2.7):
    psi_score = calculate_psi(train_data, new_data)
    return psi_score > threshold, psi_score


# -----------------------------------
# 5. Forecast Function (Next 90 Days)
# -----------------------------------

def forecast_next_90_days(model, last_day, external_data):
    """Forecast next 90 days from current day using external signals."""
    horizon = 90
    future_days = np.arange(last_day + 1, last_day + horizon + 1)
    X_future = external_data.loc[external_data['day'].isin(future_days),
    ['day', 'hotel_adr', 'airline_adr', 'airline_cxl_rate']]
    preds = np.maximum(np.round(model.predict(X_future)), 0).astype(int)
    return pd.DataFrame({
        "day": future_days,
        "forecast": preds
    })



# -----------------------------------
# 6. Clustered Forecasting with PSI-based Model Switching
# -----------------------------------

forecast_results = []
model_switch_log = []

for cluster_name, data in clusters.items():
    print(f"\n=== Processing Cluster: {cluster_name} ===")

    # Simulate "today" as day = 180
    train = data[data['day'] < 180]
    new_data = data[(data['day'] >= 180) & (data['day'] < 210)]

    X_train = train[['day', 'hotel_adr', 'airline_adr', 'airline_cxl_rate']]
    y_train = train['bookings']
    current_model_type = "linear"
    current_model = model_families[current_model_type](X_train, y_train)

    # PSI drift detection
    drift, psi_score = drift_detected_psi(train['bookings'], new_data['bookings'])
    if drift:
        new_model_type = "rf" if current_model_type == "linear" else "linear"
        print(f"PSI={psi_score:.3f} → Significant drift → Switching {current_model_type} → {new_model_type}")
        model_switch_log.append({
            "timestamp": datetime.now(),
            "cluster": cluster_name,
            "old_model": current_model_type,
            "new_model": new_model_type,
            "psi_score": psi_score
        })
        current_model_type = new_model_type
        current_model = model_families[current_model_type](X_train, y_train)
    else:
        print(f"PSI={psi_score:.3f} → Stable → Keeping {current_model_type}")

    # Forecast next 90 days
    forecast_df = forecast_next_90_days(current_model, 0, external_df)
    forecast_df["cluster"] = cluster_name
    forecast_df["model_used"] = current_model_type
    forecast_results.append(forecast_df)

# -----------------------------------
# 7. Outputs
# -----------------------------------

forecast_results = pd.concat(forecast_results, ignore_index=True)
switch_log = pd.DataFrame(model_switch_log)

print("\n=== Forecast Results (sample) ===")
print(forecast_results.head())

print("\n=== Model Switch Log ===")
print(switch_log if not switch_log.empty else "No model switches detected.")

# -----------------------------------
# 8. Visualization
# -----------------------------------

for cluster_name in clusters.keys():
    plt.figure(figsize=(8, 4))
    cluster_data = clusters[cluster_name]
    forecast_data = forecast_results[forecast_results['cluster'] == cluster_name]
    # plt.plot(cluster_data['day'], cluster_data['bookings'], label='Historical Bookings', color='gray')
    plt.plot(forecast_data['day'], forecast_data['forecast'], label=f"Forecast ({forecast_data['model_used'].iloc[0]})")
    plt.title(f"{cluster_name} Cluster — 90-Day Forecast (PSI Drift Detection)")
    plt.xlabel("Day")
    plt.ylabel("Bookings")
    plt.legend()
    plt.grid(True)
    plt.show()

# Save results
forecast_results.to_csv("forecast_results_PSI.csv", index=False)
switch_log.to_csv("model_switch_log_PSI.csv", index=False)
