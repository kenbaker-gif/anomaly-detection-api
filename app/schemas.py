from pydantic import BaseModel, Field
from typing import Optional
import numpy as np


class TransactionInput(BaseModel):
    """
    Single transaction input.
    V1-V28 are PCA-transformed features from the creditcard dataset.
    Time and Amount are the original features.
    """
    Time: float = Field(..., description="Seconds elapsed since first transaction")
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float = Field(..., description="Transaction amount in USD")

    class Config:
        json_schema_extra = {
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


class AnomalyExplanation(BaseModel):
    top_features: list[str] = Field(..., description="Top features driving the anomaly score")
    feature_contributions: dict[str, float] = Field(..., description="Feature name → deviation from training mean (in std units)")


class PredictionResponse(BaseModel):
    is_fraud: bool = Field(..., description="True if transaction is classified as anomalous/fraudulent")
    label: str = Field(..., description="'FRAUD' or 'NORMAL'")
    anomaly_score: float = Field(..., description="Raw Isolation Forest score. More negative = more anomalous.")
    confidence: float = Field(..., description="Normalized confidence score between 0.0 and 1.0")
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