from pydantic import BaseModel, Field
from typing import Optional


class TransactionInput(BaseModel):
    """
    Single transaction input.
    V1-V28 are PCA-transformed features from the creditcard dataset.
    Time and Amount are the original features.
    
    Defaults are set to 0.0 to prevent 422 errors if features are missing.
    """
    Time: float = Field(0.0, description="Seconds elapsed since first transaction")
    V1: float = 0.0; V2: float = 0.0; V3: float = 0.0; V4: float = 0.0; V5: float = 0.0
    V6: float = 0.0; V7: float = 0.0; V8: float = 0.0; V9: float = 0.0; V10: float = 0.0
    V11: float = 0.0; V12: float = 0.0; V13: float = 0.0; V14: float = 0.0; V15: float = 0.0
    V16: float = 0.0; V17: float = 0.0; V18: float = 0.0; V19: float = 0.0; V20: float = 0.0
    V21: float = 0.0; V22: float = 0.0; V23: float = 0.0; V24: float = 0.0; V25: float = 0.0
    V26: float = 0.0; V27: float = 0.0; V28: float = 0.0
    Amount: float = Field(..., description="Transaction amount in USD")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Time": 406.0,
                "V1": -2.312, "V2": 1.952, "V3": -1.610, "V4": 3.997,
                "V5": -0.522, "V6": -1.427, "V7": -2.537, "V8": 1.392,
                "V9": -2.770, "V10": -2.772, "V11": 3.202, "V12": -2.900,
                "V13": -0.595, "V14": -4.289, "V15": 0.390, "V16": -1.141,
                "V17": -2.830, "V18": -0.017, "V19": 0.417, "V20": 0.127,
                "V21": 0.517, "V22": -0.035, "V23": -0.465, "V24": 0.320,
                "V25": 0.044, "V26": -0.202, "V27": 0.472, "V28": 0.529,
                "Amount": 149.62
            }
        }
    }


class AnomalyExplanation(BaseModel):
    top_features: list[str] = Field(..., description="Top features driving the anomaly score")
    feature_contributions: dict[str, float] = Field(..., description="Feature name → deviation from training mean")


class PredictionResponse(BaseModel):
    is_fraud: bool = Field(..., description="True if transaction is classified as anomalous")
    label: str = Field(..., description="'FRAUD' or 'NORMAL'")
    anomaly_score: float = Field(..., description="Raw Isolation Forest score")
    confidence: float = Field(..., description="Normalized confidence score 0.0-1.0")
    explanation: AnomalyExplanation
    model_version: str


class BatchTransactionInput(BaseModel):
    transactions: list[TransactionInput] = Field(..., max_length=500)


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]
    total: int
    fraud_count: int
    normal_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    training_samples: Optional[int] = None


class ModelInfoResponse(BaseModel):
    model_type: str
    model_version: str
    n_estimators: int
    contamination: float
    feature_count: int
    training_samples: int
    threshold: float