from prometheus_client import Counter, Histogram, Gauge

# --- Prediction Metrics ---
PREDICTIONS_TOTAL = Counter(
    "predictions_total", 
    "Total predictions by class",
    ["prediction_class"]
)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Distribution of confidence scores",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "End-to-end prediction latency in seconds",
    ["step"] # preprocessing, inference, total
)

# --- Feature Metrics (Drift Detection) ---
FEATURE_VALUE = Gauge(
    "feature_value_latest",
    "Latest value of key features",
    ["feature_name"]
)

# --- Model Metrics ---
MODEL_INFO = Gauge(
    "model_info",
    "Model version info",
    ["version"]
)

# --- System Metrics ---
API_REQUESTS_TOTAL = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"]
)

API_ERRORS_TOTAL = Counter(
    "api_errors_total",
    "Total API errors",
    ["type"]
)
