import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, baseline_path: str = "data/baselines/training_statistics.json"):
        self.baseline_stats = self._load_baseline(baseline_path)
        
    def _load_baseline(self, path: str) -> dict:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except  FileNotFoundError:
            logger.warning(f"Baseline statistics not found at {path}")
            return {}
            
    def calculate_psi(self, expected_array, actual_array, buckets=10) -> float:
        """
        Calculate Population Stability Index (PSI)
        """
        def scale(x):
            return (x - min(expected_array)) / (max(expected_array) - min(expected_array))

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        breakpoints = np.percentile(expected_array, breakpoints)
        
        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)
        
        # Avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
        return np.sum(psi_values)

    def detect_drift(self, current_data: pd.DataFrame) -> list:
        """
        Detect drift on current batch of data
        Returns a list of drift alerts
        """
        alerts = []
        features = self.baseline_stats.get("features", {})
        
        for feature_name, stats in features.items():
            if feature_name not in current_data.columns:
                continue
                
            current_values = current_data[feature_name].dropna().values
            
            # Skip if not enough data
            if len(current_values) < 50:
                continue
            
            # Check Numerical features
            # Note: In a real implementation we would have the full training array for precise PSI/KS
            # Here we approximate or assume we have distribution buckets stored.
            # For this simplified version, we'll assume we can't run PSI without full bins.
            # We will rely on simple statistical checks if raw training data isn't available,
            # but ideally we load a sample of training data.
            
            # Statistical Check (Mean Shift)
            current_mean = np.mean(current_values)
            baseline_mean = stats.get("mean")
            baseline_std = stats.get("std")
            
            if baseline_mean is not None and baseline_std is not None:
                z_score = abs(current_mean - baseline_mean) / (baseline_std + 1e-6)
                if z_score > 2.0:
                    alerts.append({
                        "feature": feature_name,
                        "type": "mean_shift",
                        "severity": "medium",
                        "details": f"Mean shifted by {z_score:.2f} std devs"
                    })

        return alerts
