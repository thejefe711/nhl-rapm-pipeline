import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class xGValidationReport:
    brier_score: float
    log_loss: float
    roc_auc: float
    calibration: pd.DataFrame
    residual_analysis: Dict[str, Any]
    edge_case_checks: Dict[str, Any]

class xGModelValidator:
    def __init__(self, model, X_test: pd.DataFrame, y_test: pd.Series):
        self.model = model
        self.X = X_test
        self.y = y_test
        # Handle different model types (sklearn vs others)
        if hasattr(model, "predict_proba"):
            self.y_pred = model.predict_proba(X_test)[:, 1]
        else:
            self.y_pred = model.predict(X_test)
    
    def full_validation_report(self) -> xGValidationReport:
        return xGValidationReport(
            brier_score=self.brier_score(),
            log_loss=self.log_loss(),
            roc_auc=self.roc_auc(),
            calibration=self.calibration_by_decile(),
            residual_analysis=self.analyze_residuals(),
            edge_case_checks=self.check_edge_cases()
        )
    
    def brier_score(self) -> float:
        return brier_score_loss(self.y, self.y_pred)

    def log_loss(self) -> float:
        return log_loss(self.y, self.y_pred)

    def roc_auc(self) -> float:
        return roc_auc_score(self.y, self.y_pred)

    def calibration_by_decile(self) -> pd.DataFrame:
        df = pd.DataFrame({'y_true': self.y, 'y_pred': self.y_pred})
        # Handle cases with too few unique values for qcut
        try:
            df['decile'] = pd.qcut(df['y_pred'], 10, labels=False, duplicates='drop')
        except ValueError:
             df['decile'] = 0 # Fallback
             
        return df.groupby('decile').agg(
            mean_predicted=('y_pred', 'mean'),
            mean_actual=('y_true', 'mean'),
            count=('y_true', 'count')
        )
    
    def analyze_residuals(self) -> Dict[str, Any]:
        residuals = self.y - self.y_pred
        return {
            'mean_residual': residuals.mean(),
            'std_residual': residuals.std(),
            'max_residual': residuals.max(),
            'min_residual': residuals.min()
        }

    def check_edge_cases(self) -> Dict[str, Any]:
        results = {}
        # Empty net shots should have xG ~0.9
        if 'empty_net' in self.X.columns:
            empty_net_mask = self.X['empty_net'] == 1
            if empty_net_mask.any():
                empty_net_xg = self.y_pred[empty_net_mask].mean()
                results['empty_net_xg'] = {
                    'actual': float(empty_net_xg), 
                    'expected': 0.9, 
                    'tolerance': 0.1,
                    'passed': abs(empty_net_xg - 0.9) < 0.1
                }
        
        return results
