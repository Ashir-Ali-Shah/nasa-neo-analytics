import logging
import os
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MLflowService:
    """
    A minimal service wrapper for MLflow integration.
    This allows tracking experiments if MLflow is configured,
    and safely acts as a no-op if it isn't, preventing crashes.
    """
    
    def __init__(self):
        self.is_configured = False
        self.tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        
        if self.tracking_uri:
            try:
                import mlflow
                mlflow.set_tracking_uri(self.tracking_uri)
                self.is_configured = True
                logger.info(f"MLflow service initialized with tracking URI: {self.tracking_uri}")
            except ImportError:
                logger.warning("MLflow is not installed. Experiment tracking disabled.")
            except Exception as e:
                logger.error(f"Failed to initialize MLflow: {e}")
                
    def log_prediction(self, features: Dict[str, Any], prediction: Dict[str, Any], model_name: str = "xgboost_hazard_model") -> None:
        """Log a prediction event to MLflow if configured."""
        if not self.is_configured:
            return
            
        try:
            import mlflow
            # Start a run or use an existing active one
            mlflow.set_experiment("neo_hazard_predictions")
            with mlflow.start_run(run_name=f"prediction_{int(time.time())}"):
                # Log features as parameters
                for k, v in features.items():
                    mlflow.log_param(f"feature_{k}", v)
                
                # Log prediction results as metrics
                for k, v in prediction.items():
                    if isinstance(v, (int, float, bool)):
                        mlflow.log_metric(f"pred_{k}", float(v))
                        
                mlflow.set_tag("model", model_name)
        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
