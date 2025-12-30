import json
import logging
import uuid
import datetime
import os
from pathlib import Path

# Configure logger for file output
logger = logging.getLogger("prediction_logger")
logger.setLevel(logging.INFO)

# Ensure logs directory exists
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
handler = logging.FileHandler(LOG_DIR / "predictions.jsonl")
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)

class PredictionLogger:
    def __init__(self):
        self.logger = logger
    
    def log_prediction(
        self,
        features: dict,
        prediction: int,
        probabilities: list,
        confidence: float,
        latency_ms: float,
        model_version: str,
        request_id: str = None
    ):
        """
        Log prediction details to JSONL file.
        In a real production environment, this might write to a database or message queue.
        """
        if not request_id:
            request_id = str(uuid.uuid4())
            
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "request_id": request_id,
            "model_version": model_version,
            "latency_ms": latency_ms,
            "features": features,
            "prediction": prediction,
            "probabilities": list(probabilities), # Ensure serializable
            "confidence": confidence,
            "risk_level": "High" if prediction == 1 else "Low"
        }
        
        self.logger.info(json.dumps(log_entry))
        return request_id

prediction_logger = PredictionLogger()
