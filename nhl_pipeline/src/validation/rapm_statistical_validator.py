import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class RAPMValidationReport:
    coefficient_distribution: Dict[str, float]
    multicollinearity: Dict[str, float]
    residual_analysis: Dict[str, float]
    regularization_diagnostics: Dict[str, Any]
    stability: Dict[str, Any]
    predictive_validity: Dict[str, float]
    known_groups: Dict[str, bool]

class RAPMStatisticalValidator:
    def __init__(self, X, y, w, model, coefficients: np.ndarray, player_mapping: Dict[int, str]):
        self.X = X
        self.y = y
        self.w = w
        self.model = model
        self.coef = coefficients
        self.player_map = player_mapping
    
    def full_validation(self) -> RAPMValidationReport:
        return RAPMValidationReport(
            coefficient_distribution=self.check_coefficient_distribution(),
            multicollinearity=self.check_multicollinearity(),
            residual_analysis=self.analyze_residuals(),
            regularization_diagnostics=self.check_regularization(),
            stability=self.check_stability(),
            predictive_validity=self.check_predictive_validity(),
            known_groups=self.check_known_groups()
        )
    
    def check_coefficient_distribution(self) -> Dict[str, float]:
        return {
            'mean': float(np.mean(self.coef)),
            'std': float(np.std(self.coef)),
            'min': float(np.min(self.coef)),
            'max': float(np.max(self.coef)),
            'skew': float(pd.Series(self.coef).skew()),
            'kurtosis': float(pd.Series(self.coef).kurtosis())
        }

    def check_multicollinearity(self) -> Dict[str, float]:
        # Condition number approximation (expensive for large sparse matrices, so maybe skip or use approx)
        # For now, placeholder
        return {'condition_number': 0.0} 

    def analyze_residuals(self) -> Dict[str, float]:
        y_pred = self.model.predict(self.X)
        residuals = self.y - y_pred
        return {
            'mean_residual': float(np.mean(residuals)),
            'std_residual': float(np.std(residuals))
        }

    def check_regularization(self) -> Dict[str, Any]:
        return {
            'alpha': float(self.model.alpha) if hasattr(self.model, 'alpha') else getattr(self.model, 'alpha_', 0.0)
        }

    def check_stability(self) -> Dict[str, Any]:
        bootstrap_df = self.check_stability_bootstrap(n_bootstrap=5) # 5 for speed in dev
        if bootstrap_df.empty:
            return {'status': 'failed'}
        return {
            'mean_std': float(bootstrap_df['coef_std'].mean()),
            'max_std': float(bootstrap_df['coef_std'].max())
        }

    def check_predictive_validity(self) -> Dict[str, float]:
        # Placeholder
        return {'r_squared': 0.0}

    def check_known_groups(self) -> Dict[str, bool]:
        # Placeholder
        return {'elite_check': True}

    def check_stability_bootstrap(self, n_bootstrap=10) -> pd.DataFrame:
        """Bootstrap confidence intervals for each coefficient"""
        # Reduced n_bootstrap for performance in this snippet
        n_obs = self.X.shape[0]
        bootstrap_coefs = []
        
        # This is computationally expensive, so use with caution
        try:
            for i in range(n_bootstrap):
                idx = np.random.choice(n_obs, size=n_obs, replace=True)
                X_boot = self.X[idx]
                y_boot = self.y[idx]
                w_boot = self.w[idx]
                
                # Re-fit model
                model = Ridge(alpha=getattr(self.model, 'alpha', 100.0))
                model.fit(X_boot, y_boot, sample_weight=w_boot)
                bootstrap_coefs.append(model.coef_)
            
            bootstrap_coefs = np.array(bootstrap_coefs)
            return pd.DataFrame({
                'player_id': list(self.player_map.keys()),
                'coef_mean': bootstrap_coefs.mean(axis=0),
                'coef_std': bootstrap_coefs.std(axis=0),
                'ci_lower': np.percentile(bootstrap_coefs, 2.5, axis=0),
                'ci_upper': np.percentile(bootstrap_coefs, 97.5, axis=0)
            })
        except Exception as e:
            print(f"Bootstrap failed: {e}")
            return pd.DataFrame()
