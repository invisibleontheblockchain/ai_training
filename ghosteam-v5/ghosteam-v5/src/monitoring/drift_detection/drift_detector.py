import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DriftDetector:
    """Comprehensive drift detection for ML models using multiple statistical methods."""
    
    def __init__(self, 
                 methods: List[str] = ["ks", "mmd", "chi2", "psi"],
                 threshold: float = 0.05,
                 min_samples: int = 100):
        """Initialize drift detector.
        
        Args:
            methods: List of drift detection methods to use
            threshold: Statistical significance threshold
            min_samples: Minimum samples required for drift detection
        """
        self.methods = methods
        self.threshold = threshold
        self.min_samples = min_samples
        self.reference_data = None
        self.reference_stats = {}
        self.scaler = StandardScaler()
        self.pca = None
        
    def fit(self, reference_data: pd.DataFrame) -> None:
        """Fit drift detector on reference data.
        
        Args:
            reference_data: Reference dataset to compare against
        """
        if len(reference_data) < self.min_samples:
            raise ValueError(f"Reference data must have at least {self.min_samples} samples")
        
        self.reference_data = reference_data.copy()
        
        # Compute reference statistics
        self._compute_reference_stats()
        
        # Fit preprocessing components
        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.scaler.fit(self.reference_data[numeric_cols])
            
            # Fit PCA for dimensionality reduction if many features
            if len(numeric_cols) > 10:
                scaled_data = self.scaler.transform(self.reference_data[numeric_cols])
                self.pca = PCA(n_components=min(10, len(numeric_cols)))
                self.pca.fit(scaled_data)
        
        logger.info(f"Drift detector fitted on {len(reference_data)} samples with {len(reference_data.columns)} features")
    
    def _compute_reference_stats(self) -> None:
        """Compute reference statistics for each feature."""
        self.reference_stats = {}
        
        for column in self.reference_data.columns:
            col_data = self.reference_data[column].dropna()
            
            if pd.api.types.is_numeric_dtype(col_data):
                # Numeric feature statistics
                self.reference_stats[column] = {
                    "type": "numeric",
                    "mean": col_data.mean(),
                    "std": col_data.std(),
                    "min": col_data.min(),
                    "max": col_data.max(),
                    "q25": col_data.quantile(0.25),
                    "q50": col_data.quantile(0.50),
                    "q75": col_data.quantile(0.75),
                    "skewness": col_data.skew(),
                    "kurtosis": col_data.kurtosis()
                }
            else:
                # Categorical feature statistics
                value_counts = col_data.value_counts(normalize=True)
                self.reference_stats[column] = {
                    "type": "categorical",
                    "value_counts": value_counts.to_dict(),
                    "unique_count": col_data.nunique(),
                    "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None,
                    "entropy": -sum(p * np.log2(p) for p in value_counts.values if p > 0)
                }
    
    def detect_drift(self, 
                    current_data: pd.DataFrame,
                    features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Detect drift in current data compared to reference.
        
        Args:
            current_data: Current dataset to check for drift
            features: Specific features to check (if None, check all)
            
        Returns:
            Comprehensive drift detection results
        """
        if self.reference_data is None:
            raise ValueError("Must fit detector on reference data first")
        
        if len(current_data) < self.min_samples:
            logger.warning(f"Current data has only {len(current_data)} samples, minimum {self.min_samples} recommended")
        
        if features is None:
            features = [col for col in current_data.columns if col in self.reference_data.columns]
        
        results = {
            "timestamp": datetime.now(),
            "drift_detected": False,
            "overall_drift_score": 0.0,
            "feature_drift": {},
            "method_results": {},
            "summary": {
                "total_features": len(features),
                "drifted_features": 0,
                "drift_methods_triggered": []
            }
        }
        
        # Check drift for each feature
        for feature in features:
            if feature not in self.reference_data.columns:
                logger.warning(f"Feature {feature} not found in reference data")
                continue
            
            feature_results = self._detect_feature_drift(
                feature, 
                self.reference_data[feature], 
                current_data[feature]
            )
            
            results["feature_drift"][feature] = feature_results
            
            # Update overall drift status
            if feature_results.get("drift_detected", False):
                results["drift_detected"] = True
                results["summary"]["drifted_features"] += 1
                
                # Track which methods detected drift
                for method, method_result in feature_results.items():
                    if isinstance(method_result, dict) and method_result.get("drift_detected", False):
                        if method not in results["summary"]["drift_methods_triggered"]:
                            results["summary"]["drift_methods_triggered"].append(method)
            
            # Update overall drift score
            feature_score = feature_results.get("drift_score", 0.0)
            results["overall_drift_score"] = max(results["overall_drift_score"], feature_score)
        
        # Multivariate drift detection
        if len(features) > 1:
            multivariate_results = self._detect_multivariate_drift(current_data, features)
            results["multivariate_drift"] = multivariate_results
            
            if multivariate_results.get("drift_detected", False):
                results["drift_detected"] = True
                results["overall_drift_score"] = max(
                    results["overall_drift_score"], 
                    multivariate_results.get("drift_score", 0.0)
                )
        
        # Calculate drift severity
        results["drift_severity"] = self._calculate_drift_severity(results["overall_drift_score"])
        
        return results
    
    def _detect_feature_drift(self, 
                             feature_name: str,
                             ref_values: pd.Series, 
                             cur_values: pd.Series) -> Dict[str, Any]:
        """Detect drift for a single feature.
        
        Args:
            feature_name: Name of the feature
            ref_values: Reference values
            cur_values: Current values
            
        Returns:
            Feature-level drift results
        """
        feature_results = {
            "drift_detected": False,
            "drift_score": 0.0,
            "feature_type": self.reference_stats[feature_name]["type"]
        }
        
        # Clean data
        ref_clean = ref_values.dropna()
        cur_clean = cur_values.dropna()
        
        if len(ref_clean) == 0 or len(cur_clean) == 0:
            logger.warning(f"No valid data for feature {feature_name}")
            return feature_results
        
        # Apply appropriate drift detection methods based on feature type
        if self.reference_stats[feature_name]["type"] == "numeric":
            feature_results.update(self._detect_numeric_drift(ref_clean, cur_clean))
        else:
            feature_results.update(self._detect_categorical_drift(ref_clean, cur_clean))
        
        # Determine overall drift for this feature
        drift_detected = any(
            method_result.get("drift_detected", False) 
            for method_result in feature_results.values()
            if isinstance(method_result, dict)
        )
        
        feature_results["drift_detected"] = drift_detected
        feature_results["drift_score"] = self._calculate_feature_drift_score(feature_results)
        
        return feature_results
    
    def _detect_numeric_drift(self, ref_values: pd.Series, cur_values: pd.Series) -> Dict[str, Any]:
        """Detect drift for numeric features.
        
        Args:
            ref_values: Reference numeric values
            cur_values: Current numeric values
            
        Returns:
            Numeric drift detection results
        """
        results = {}
        
        # Kolmogorov-Smirnov test
        if "ks" in self.methods:
            ks_stat, ks_pvalue = stats.ks_2samp(ref_values, cur_values)
            results["ks"] = {
                "statistic": float(ks_stat),
                "p_value": float(ks_pvalue),
                "drift_detected": ks_pvalue < self.threshold,
                "interpretation": "Distribution shape difference"
            }
        
        # Maximum Mean Discrepancy
        if "mmd" in self.methods:
            mmd_score = self._compute_mmd(ref_values.values, cur_values.values)
            mmd_threshold = 0.1  # Configurable threshold
            results["mmd"] = {
                "score": float(mmd_score),
                "threshold": mmd_threshold,
                "drift_detected": mmd_score > mmd_threshold,
                "interpretation": "Mean embedding difference"
            }
        
        # Wasserstein distance
        if "wasserstein" in self.methods:
            wasserstein_dist = stats.wasserstein_distance(ref_values, cur_values)
            # Normalize by reference standard deviation
            ref_std = ref_values.std()
            normalized_dist = wasserstein_dist / ref_std if ref_std > 0 else wasserstein_dist
            results["wasserstein"] = {
                "distance": float(wasserstein_dist),
                "normalized_distance": float(normalized_dist),
                "drift_detected": normalized_dist > 0.1,
                "interpretation": "Earth mover's distance"
            }
        
        # Population Stability Index (PSI)
        if "psi" in self.methods:
            psi_score = self._compute_psi_numeric(ref_values, cur_values)
            results["psi"] = {
                "score": float(psi_score),
                "drift_detected": psi_score > 0.2,  # Standard PSI threshold
                "interpretation": "Population stability"
            }
        
        return results
    
    def _detect_categorical_drift(self, ref_values: pd.Series, cur_values: pd.Series) -> Dict[str, Any]:
        """Detect drift for categorical features.
        
        Args:
            ref_values: Reference categorical values
            cur_values: Current categorical values
            
        Returns:
            Categorical drift detection results
        """
        results = {}
        
        # Chi-square test
        if "chi2" in self.methods:
            chi2_stat, chi2_pvalue = self._chi2_test(ref_values, cur_values)
            results["chi2"] = {
                "statistic": float(chi2_stat),
                "p_value": float(chi2_pvalue),
                "drift_detected": chi2_pvalue < self.threshold,
                "interpretation": "Category distribution difference"
            }
        
        # Population Stability Index for categorical
        if "psi" in self.methods:
            psi_score = self._compute_psi_categorical(ref_values, cur_values)
            results["psi"] = {
                "score": float(psi_score),
                "drift_detected": psi_score > 0.2,
                "interpretation": "Population stability"
            }
        
        # Jensen-Shannon divergence
        if "js" in self.methods:
            js_div = self._compute_js_divergence(ref_values, cur_values)
            results["js_divergence"] = {
                "divergence": float(js_div),
                "drift_detected": js_div > 0.1,
                "interpretation": "Probability distribution divergence"
            }
        
        return results
    
    def _detect_multivariate_drift(self, current_data: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """Detect multivariate drift using dimensionality reduction.
        
        Args:
            current_data: Current dataset
            features: Features to analyze
            
        Returns:
            Multivariate drift results
        """
        try:
            # Get numeric features only for multivariate analysis
            numeric_features = [f for f in features if f in self.reference_data.select_dtypes(include=[np.number]).columns]
            
            if len(numeric_features) < 2:
                return {"drift_detected": False, "reason": "Insufficient numeric features"}
            
            # Prepare data
            ref_data = self.reference_data[numeric_features].dropna()
            cur_data = current_data[numeric_features].dropna()
            
            if len(ref_data) == 0 or len(cur_data) == 0:
                return {"drift_detected": False, "reason": "No valid data"}
            
            # Scale data
            ref_scaled = self.scaler.transform(ref_data)
            cur_scaled = self.scaler.transform(cur_data)
            
            # Apply PCA if fitted
            if self.pca is not None:
                ref_pca = self.pca.transform(ref_scaled)
                cur_pca = self.pca.transform(cur_scaled)
            else:
                ref_pca = ref_scaled
                cur_pca = cur_scaled
            
            # Compute multivariate MMD
            mmd_score = self._compute_mmd_multivariate(ref_pca, cur_pca)
            
            return {
                "drift_detected": mmd_score > 0.1,
                "mmd_score": float(mmd_score),
                "features_analyzed": numeric_features,
                "interpretation": "Multivariate distribution shift"
            }
            
        except Exception as e:
            logger.warning(f"Multivariate drift detection failed: {e}")
            return {"drift_detected": False, "error": str(e)}
    
    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
        """Compute Maximum Mean Discrepancy using RBF kernel.
        
        Args:
            X: Reference samples
            Y: Current samples
            gamma: RBF kernel parameter
            
        Returns:
            MMD score
        """
        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1, 1)
        
        XX = rbf_kernel(X, X, gamma=gamma)
        YY = rbf_kernel(Y, Y, gamma=gamma)
        XY = rbf_kernel(X, Y, gamma=gamma)
        
        mmd = XX.mean() + YY.mean() - 2 * XY.mean()
        return max(0, mmd)
    
    def _compute_mmd_multivariate(self, X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
        """Compute multivariate MMD.
        
        Args:
            X: Reference samples (n_samples, n_features)
            Y: Current samples (n_samples, n_features)
            gamma: RBF kernel parameter
            
        Returns:
            Multivariate MMD score
        """
        XX = rbf_kernel(X, X, gamma=gamma)
        YY = rbf_kernel(Y, Y, gamma=gamma)
        XY = rbf_kernel(X, Y, gamma=gamma)
        
        mmd = XX.mean() + YY.mean() - 2 * XY.mean()
        return max(0, mmd)
    
    def _chi2_test(self, ref_values: pd.Series, cur_values: pd.Series) -> Tuple[float, float]:
        """Chi-square test for categorical features.
        
        Args:
            ref_values: Reference categorical values
            cur_values: Current categorical values
            
        Returns:
            Chi-square statistic and p-value
        """
        # Get value counts for both distributions
        ref_counts = ref_values.value_counts()
        cur_counts = cur_values.value_counts()
        
        # Align categories
        all_categories = set(ref_counts.index) | set(cur_counts.index)
        ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
        cur_aligned = [cur_counts.get(cat, 0) for cat in all_categories]
        
        # Add small constant to avoid zero counts
        ref_aligned = [max(1, count) for count in ref_aligned]
        cur_aligned = [max(1, count) for count in cur_aligned]
        
        # Perform chi-square test
        chi2_stat, p_value = stats.chisquare(cur_aligned, ref_aligned)
        return chi2_stat, p_value
    
    def _compute_psi_numeric(self, ref_values: pd.Series, cur_values: pd.Series, bins: int = 10) -> float:
        """Compute Population Stability Index for numeric features.
        
        Args:
            ref_values: Reference values
            cur_values: Current values
            bins: Number of bins for discretization
            
        Returns:
            PSI score
        """
        # Create bins based on reference data
        bin_edges = np.histogram_bin_edges(ref_values, bins=bins)
        
        # Get bin counts
        ref_counts, _ = np.histogram(ref_values, bins=bin_edges)
        cur_counts, _ = np.histogram(cur_values, bins=bin_edges)
        
        # Convert to proportions
        ref_props = ref_counts / ref_counts.sum()
        cur_props = cur_counts / cur_counts.sum()
        
        # Add small constant to avoid log(0)
        ref_props = np.where(ref_props == 0, 1e-6, ref_props)
        cur_props = np.where(cur_props == 0, 1e-6, cur_props)
        
        # Calculate PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        return psi
    
    def _compute_psi_categorical(self, ref_values: pd.Series, cur_values: pd.Series) -> float:
        """Compute PSI for categorical features.
        
        Args:
            ref_values: Reference categorical values
            cur_values: Current categorical values
            
        Returns:
            PSI score
        """
        ref_props = ref_values.value_counts(normalize=True)
        cur_props = cur_values.value_counts(normalize=True)
        
        # Align categories
        all_categories = set(ref_props.index) | set(cur_props.index)
        
        psi = 0.0
        for cat in all_categories:
            ref_prop = ref_props.get(cat, 1e-6)
            cur_prop = cur_props.get(cat, 1e-6)
            psi += (cur_prop - ref_prop) * np.log(cur_prop / ref_prop)
        
        return psi
    
    def _compute_js_divergence(self, ref_values: pd.Series, cur_values: pd.Series) -> float:
        """Compute Jensen-Shannon divergence for categorical features.
        
        Args:
            ref_values: Reference categorical values
            cur_values: Current categorical values
            
        Returns:
            JS divergence
        """
        ref_props = ref_values.value_counts(normalize=True)
        cur_props = cur_values.value_counts(normalize=True)
        
        # Align categories
        all_categories = set(ref_props.index) | set(cur_props.index)
        
        P = np.array([ref_props.get(cat, 1e-6) for cat in all_categories])
        Q = np.array([cur_props.get(cat, 1e-6) for cat in all_categories])
        
        # Normalize
        P = P / P.sum()
        Q = Q / Q.sum()
        
        # Compute JS divergence
        M = 0.5 * (P + Q)
        js_div = 0.5 * stats.entropy(P, M) + 0.5 * stats.entropy(Q, M)
        
        return js_div
    
    def _calculate_feature_drift_score(self, feature_results: Dict[str, Any]) -> float:
        """Calculate aggregated drift score for a feature.
        
        Args:
            feature_results: Feature drift detection results
            
        Returns:
            Aggregated drift score
        """
        scores = []
        
        for method, result in feature_results.items():
            if isinstance(result, dict):
                if "statistic" in result:
                    scores.append(result["statistic"])
                elif "score" in result:
                    scores.append(result["score"])
                elif "distance" in result:
                    scores.append(result["distance"])
                elif "divergence" in result:
                    scores.append(result["divergence"])
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_drift_severity(self, drift_score: float) -> str:
        """Calculate drift severity level.
        
        Args:
            drift_score: Overall drift score
            
        Returns:
            Severity level string
        """
        if drift_score < 0.1:
            return "low"
        elif drift_score < 0.3:
            return "medium"
        elif drift_score < 0.5:
            return "high"
        else:
            return "critical"
