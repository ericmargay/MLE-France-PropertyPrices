"""
Robust Machine Learning Price Estimation Module
================================================

A statistically sound approach to estimating current property prices
from historical transaction data using Machine Learning.

METHODOLOGY OVERVIEW
====================

This module uses a THREE-COMPONENT approach:

┌─────────────────────────────────────────────────────────────────────┐
│                    PRICE ESTIMATION PIPELINE                         │
│                                                                      │
│   1. SPATIAL MODEL (LightGBM)                                       │
│      → Learns price patterns across geography                        │
│      → Features: coordinates, property type, surface, local density  │
│      → Output: Expected price given location and property features   │
│                                                                      │
│   2. TEMPORAL ADJUSTMENT (Trend Extrapolation)                      │
│      → Estimates price trend from historical data                    │
│      → Uses linear regression on log-prices over time                │
│      → Extrapolates trend to prediction date                         │
│                                                                      │
│   3. HIERARCHICAL SMOOTHING (Bayesian-inspired)                     │
│      → Combines zone estimate with parent region estimate            │
│      → Weight based on local data reliability                        │
│      → Handles sparse data by borrowing from larger regions          │
│                                                                      │
│   Final Estimate = Spatial_Prediction × Temporal_Adjustment          │
│                    (smoothed with hierarchical prior)                │
└─────────────────────────────────────────────────────────────────────┘

WHY THIS APPROACH?
==================

1. SPATIAL MODEL (Cross-sectional)
   - LightGBM excels at learning complex non-linear patterns in space
   - Captures: expensive vs cheap areas, urban vs rural, coastal premium
   - Does NOT extrapolate in time (trees can't extrapolate)

2. TEMPORAL ADJUSTMENT (Time-series)
   - Simple linear trend on log-prices is robust and interpretable
   - Handles the extrapolation that tree models cannot do
   - Based on observed price changes in the data period

3. HIERARCHICAL SMOOTHING
   - Small zones (few transactions) have high variance
   - We "shrink" unreliable estimates toward regional average
   - This is a simplified empirical Bayes approach
   - Reduces extreme predictions in sparse areas

CONFIDENCE SCORING
==================
Based on three factors:
- Model uncertainty (prediction variance)
- Data density (transaction count in zone)  
- Data recency (how recent are the transactions)

REFERENCES
==========
- Spatial hedonic models: Bourassa et al. (2007)
- Hierarchical modeling: Gelman & Hill (2006)
- LightGBM: Ke et al. (2017)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import lightgbm as lgb
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Reference dates
DATA_END_DATE = datetime(2023, 12, 31)      # End of transaction data
PREDICTION_DATE = datetime(2025, 1, 1)       # Target prediction date

# Model parameters
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

# Hierarchical smoothing parameters
MIN_TRANSACTIONS_FULL_WEIGHT = 50    # Zone with >= this many tx gets full weight
MIN_TRANSACTIONS_FOR_ESTIMATE = 3   # Minimum tx to make any estimate


class RobustPriceModel:
    """
    Robust ML model for property price estimation.
    
    Combines:
    1. Spatial model (LightGBM) for geographic price patterns
    2. Temporal model (Linear trend) for time adjustment
    3. Hierarchical smoothing for sparse data handling
    """
    
    def __init__(self, prediction_date=PREDICTION_DATE):
        self.prediction_date = prediction_date
        self.spatial_model = None
        self.temporal_model = None
        self.label_encoders = {}
        self.feature_columns = None
        self.is_fitted = False
        
        # Training statistics
        self.global_median_price = None
        self.global_std_price = None
        self.annual_trend = None  # Annual price change rate
        self.department_medians = {}  # For hierarchical smoothing
        self.region_medians = {}
        
        # Model diagnostics
        self.cv_rmse = None
        self.cv_r2 = None
        self.feature_importance = None
    
    def _engineer_features(self, df, is_training=True):
        """
        Engineer features for the spatial model.
        
        Features:
        ---------
        Spatial:
            - latitude, longitude (continuous)
            - lat_lon_interaction (captures diagonal patterns)
            - dept_encoded (categorical)
        
        Property:
            - log_surface (log of living area)
            - type_encoded (apartment, house, etc.)
        
        Local Market:
            - local_density (transactions per km² in vicinity)
        
        Note: We do NOT include temporal features here.
        The spatial model is cross-sectional (learns geography).
        Time adjustment is handled separately.
        """
        features = pd.DataFrame(index=df.index)
        
        # --- SPATIAL FEATURES ---
        if 'latitude' in df.columns:
            features['latitude'] = df['latitude'].fillna(46.6)
            features['longitude'] = df['longitude'].fillna(2.5)
        elif hasattr(df, 'geometry'):
            features['longitude'] = df.geometry.x
            features['latitude'] = df.geometry.y
        else:
            features['latitude'] = 46.6
            features['longitude'] = 2.5
        
        # Interaction term (captures NE-SW vs NW-SE patterns)
        features['lat_lon_interaction'] = features['latitude'] * features['longitude']
        
        # Squared terms (captures non-linear spatial effects)
        features['lat_squared'] = features['latitude'] ** 2
        features['lon_squared'] = features['longitude'] ** 2
        
        # Department encoding
        if 'code_departement' in df.columns:
            dept = df['code_departement'].astype(str).str.zfill(2)
        elif 'code_commune' in df.columns:
            dept = df['code_commune'].astype(str).str[:2]
        else:
            dept = pd.Series(['00'] * len(df), index=df.index)
        
        if is_training:
            self.label_encoders['dept'] = LabelEncoder()
            features['dept_encoded'] = self.label_encoders['dept'].fit_transform(dept)
        else:
            known = set(self.label_encoders['dept'].classes_)
            dept_safe = dept.apply(lambda x: x if x in known else '00')
            features['dept_encoded'] = self.label_encoders['dept'].transform(dept_safe)
        
        # --- PROPERTY FEATURES ---
        if 'surface_reelle_bati' in df.columns:
            surface = df['surface_reelle_bati'].fillna(df['surface_reelle_bati'].median())
            features['log_surface'] = np.log1p(surface.clip(lower=10, upper=500))
        else:
            features['log_surface'] = np.log1p(80)
        
        if 'type_local' in df.columns:
            prop_type = df['type_local'].fillna('Appartement')
            if is_training:
                self.label_encoders['type'] = LabelEncoder()
                features['type_encoded'] = self.label_encoders['type'].fit_transform(prop_type)
            else:
                known = set(self.label_encoders['type'].classes_)
                type_safe = prop_type.apply(lambda x: x if x in known else 'Appartement')
                features['type_encoded'] = self.label_encoders['type'].transform(type_safe)
        else:
            features['type_encoded'] = 0
        
        return features
    
    def _estimate_temporal_trend(self, df):
        """
        Estimate annual price trend using linear regression on log-prices.
        
        Model: log(price) = α + β × year + ε
        
        The coefficient β represents the annual growth rate (approximately).
        
        This is more robust than compound growth assumption because:
        1. It's estimated from actual data, not assumed
        2. Linear regression is robust to outliers when using log-prices
        3. We can assess confidence in the trend estimate
        """
        print("  Estimating temporal price trend...")
        
        df_temp = df.copy()
        df_temp['date'] = pd.to_datetime(df_temp['date_mutation'])
        df_temp['year_decimal'] = df_temp['date'].dt.year + df_temp['date'].dt.dayofyear / 365.25
        df_temp['log_price'] = np.log(df_temp['price_per_m2'])
        
        # Remove outliers for trend estimation
        q01 = df_temp['log_price'].quantile(0.01)
        q99 = df_temp['log_price'].quantile(0.99)
        df_clean = df_temp[(df_temp['log_price'] >= q01) & (df_temp['log_price'] <= q99)]
        
        # Fit linear trend
        X = df_clean[['year_decimal']].values
        y = df_clean['log_price'].values
        
        self.temporal_model = LinearRegression()
        self.temporal_model.fit(X, y)
        
        # Annual trend (coefficient)
        self.annual_trend = self.temporal_model.coef_[0]
        
        # Convert to percentage
        annual_pct = (np.exp(self.annual_trend) - 1) * 100
        
        print(f"    Estimated annual trend: {annual_pct:+.2f}%")
        print(f"    (Based on {len(df_clean):,} transactions)")
        
        return self.annual_trend
    
    def _compute_temporal_adjustment(self, transaction_date):
        """
        Compute adjustment factor to bring price to prediction date.
        
        Factor = exp(β × years_forward)
        
        where β is the annual trend coefficient.
        """
        if self.temporal_model is None:
            return 1.0
        
        if pd.isna(transaction_date):
            return 1.0
        
        tx_date = pd.to_datetime(transaction_date)
        tx_year = tx_date.year + tx_date.dayofyear / 365.25
        pred_year = self.prediction_date.year + self.prediction_date.timetuple().tm_yday / 365.25
        
        years_forward = pred_year - tx_year
        
        # Apply trend
        adjustment = np.exp(self.annual_trend * years_forward)
        
        return adjustment
    
    def fit(self, transactions_df, target_col='price_per_m2'):
        """
        Train the robust price model.
        
        Steps:
        1. Compute global statistics and hierarchical priors
        2. Estimate temporal trend
        3. Train spatial model (LightGBM)
        4. Evaluate with cross-validation
        """
        print("\n" + "="*60)
        print("TRAINING ROBUST ML PRICE MODEL")
        print("="*60)
        
        # Clean data
        df = transactions_df.copy()
        df = df[df[target_col].notna() & (df[target_col] > 100) & (df[target_col] < 50000)]
        
        print(f"Training samples: {len(df):,}")
        
        # =================================================================
        # STEP 1: Compute global and hierarchical statistics
        # =================================================================
        print("\nStep 1: Computing hierarchical priors...")
        
        self.global_median_price = df[target_col].median()
        self.global_std_price = df[target_col].std()
        
        print(f"  Global median: €{self.global_median_price:,.0f}/m²")
        print(f"  Global std: €{self.global_std_price:,.0f}/m²")
        
        # Department medians (for hierarchical smoothing)
        if 'code_departement' in df.columns:
            dept_stats = df.groupby('code_departement')[target_col].agg(['median', 'count'])
            self.department_medians = dept_stats['median'].to_dict()
            print(f"  Computed medians for {len(self.department_medians)} departments")
        
        # =================================================================
        # STEP 2: Estimate temporal trend
        # =================================================================
        print("\nStep 2: Estimating temporal trend...")
        self._estimate_temporal_trend(df)
        
        # =================================================================
        # STEP 3: Prepare features and target (with temporal adjustment)
        # =================================================================
        print("\nStep 3: Engineering features...")
        
        X = self._engineer_features(df, is_training=True)
        self.feature_columns = X.columns.tolist()
        
        # Adjust target prices to prediction date
        # This way, the spatial model learns the "current" price surface
        print("  Adjusting prices to prediction date...")
        
        df['temporal_adj'] = df['date_mutation'].apply(self._compute_temporal_adjustment)
        y_adjusted = df[target_col] * df['temporal_adj']
        
        # Log transform for better distribution
        y = np.log(y_adjusted)
        
        print(f"  Features: {len(self.feature_columns)}")
        
        # =================================================================
        # STEP 4: Train spatial model with cross-validation
        # =================================================================
        print("\nStep 4: Training spatial model (LightGBM)...")
        
        # Cross-validation to assess model quality
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_predictions = cross_val_predict(
            lgb.LGBMRegressor(**LGBM_PARAMS, n_estimators=300),
            X, y, cv=kfold, n_jobs=-1
        )
        
        # CV metrics
        cv_rmse_log = np.sqrt(np.mean((cv_predictions - y) ** 2))
        cv_r2 = 1 - np.sum((y - cv_predictions) ** 2) / np.sum((y - y.mean()) ** 2)
        
        # Convert RMSE to percentage (approximate)
        cv_mape = np.mean(np.abs(np.exp(cv_predictions) - np.exp(y)) / np.exp(y)) * 100
        
        self.cv_rmse = cv_rmse_log
        self.cv_r2 = cv_r2
        
        print(f"  Cross-validation R²: {cv_r2:.3f}")
        print(f"  Cross-validation MAPE: {cv_mape:.1f}%")
        
        # Train final model on all data
        print("  Training final model...")
        
        train_data = lgb.Dataset(X, label=y)
        
        self.spatial_model = lgb.train(
            LGBM_PARAMS,
            train_data,
            num_boost_round=300,
        )
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.spatial_model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        print("\n  Top features by importance:")
        for _, row in self.feature_importance.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.0f}")
        
        self.is_fitted = True
        
        print("\n" + "="*60)
        print("✓ MODEL TRAINING COMPLETE")
        print("="*60)
        
        return self
    
    def predict_zone(self, zone_transactions, zone_centroid=None, 
                     parent_estimate=None, parent_confidence=0.5):
        """
        Predict current price for a geographic zone.
        
        Methodology:
        1. Use spatial model to predict price for each transaction
        2. Aggregate predictions (median)
        3. Apply hierarchical smoothing with parent region
        4. Compute confidence score
        
        Parameters:
        -----------
        zone_transactions : pd.DataFrame
            Historical transactions in this zone
        zone_centroid : tuple (lon, lat)
            Zone center (used if no transactions)
        parent_estimate : float
            Price estimate from parent region (for hierarchical smoothing)
        parent_confidence : float
            Weight to give parent estimate (0-1)
        
        Returns:
        --------
        dict with prediction results
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n_transactions = len(zone_transactions)
        
        # Handle empty zones
        if n_transactions == 0:
            return {
                'predicted_price': parent_estimate if parent_estimate else self.global_median_price,
                'median_price': parent_estimate if parent_estimate else self.global_median_price,
                'weighted_price': parent_estimate if parent_estimate else self.global_median_price,
                'lower_bound': np.nan,
                'upper_bound': np.nan,
                'std_dev': 0.0,
                'transaction_count': 0,
                'confidence_score': 0,
                'estimate_quality': 'No Data',
                'last_transaction_date': None
            }
        
        # =================================================================
        # STEP 1: Generate predictions using spatial model
        # =================================================================
        X_zone = self._engineer_features(zone_transactions, is_training=False)
        
        # Predict log-prices (already adjusted to prediction date during training)
        log_predictions = self.spatial_model.predict(X_zone)
        predictions = np.exp(log_predictions)
        
        # Zone estimate: median of predictions
        zone_prediction = np.median(predictions)
        zone_std = np.std(predictions)
        
        # Prediction interval (based on prediction distribution)
        lower_pred = np.percentile(predictions, 10)
        upper_pred = np.percentile(predictions, 90)
        
        # =================================================================
        # STEP 2: Hierarchical smoothing
        # =================================================================
        # For zones with few transactions, shrink toward parent estimate
        
        if parent_estimate is not None and n_transactions < MIN_TRANSACTIONS_FULL_WEIGHT:
            # Weight based on sample size (more data = more weight on local estimate)
            local_weight = min(1.0, n_transactions / MIN_TRANSACTIONS_FULL_WEIGHT)
            
            # Blend local and parent estimates
            smoothed_prediction = (
                local_weight * zone_prediction + 
                (1 - local_weight) * parent_estimate
            )
            
            # Also smooth the bounds
            lower_smoothed = (
                local_weight * lower_pred + 
                (1 - local_weight) * (parent_estimate * 0.85)
            )
            upper_smoothed = (
                local_weight * upper_pred + 
                (1 - local_weight) * (parent_estimate * 1.15)
            )
        else:
            smoothed_prediction = zone_prediction
            lower_smoothed = lower_pred
            upper_smoothed = upper_pred
        
        # =================================================================
        # STEP 3: Compute confidence score
        # =================================================================
        
        # Model uncertainty score (0-40): based on prediction interval width
        relative_width = (upper_smoothed - lower_smoothed) / smoothed_prediction
        if relative_width < 0.15:
            uncertainty_score = 40
        elif relative_width < 0.25:
            uncertainty_score = 35
        elif relative_width < 0.35:
            uncertainty_score = 30
        elif relative_width < 0.45:
            uncertainty_score = 25
        elif relative_width < 0.55:
            uncertainty_score = 18
        elif relative_width < 0.70:
            uncertainty_score = 10
        else:
            uncertainty_score = 5
        
        # Volume score (0-40)
        if n_transactions >= 100:
            volume_score = 40
        elif n_transactions >= 50:
            volume_score = 35
        elif n_transactions >= 30:
            volume_score = 30
        elif n_transactions >= 20:
            volume_score = 25
        elif n_transactions >= 10:
            volume_score = 15
        elif n_transactions >= 5:
            volume_score = 8
        else:
            volume_score = 5
        
        # Freshness score (0-20)
        last_date = pd.to_datetime(zone_transactions['date_mutation']).max()
        days_to_pred = (self.prediction_date - last_date).days
        
        if days_to_pred <= 180:
            freshness_score = 20
        elif days_to_pred <= 365:
            freshness_score = 16
        elif days_to_pred <= 545:
            freshness_score = 12
        elif days_to_pred <= 730:
            freshness_score = 8
        else:
            freshness_score = 4
        
        confidence_score = int(uncertainty_score + volume_score + freshness_score)
        
        # Quality label
        if confidence_score >= 85:
            quality = 'Very High'
        elif confidence_score >= 70:
            quality = 'High'
        elif confidence_score >= 55:
            quality = 'Medium'
        elif confidence_score >= 35:
            quality = 'Low'
        else:
            quality = 'Very Low'
        
        # Historical median (for comparison)
        historical_median = zone_transactions['price_per_m2'].median()
        
        return {
            'predicted_price': float(smoothed_prediction),
            'median_price': float(smoothed_prediction),      # Alias
            'weighted_price': float(smoothed_prediction),    # Alias
            'lower_bound': float(lower_smoothed),
            'upper_bound': float(upper_smoothed),
            'std_dev': float(zone_std),
            'transaction_count': int(n_transactions),
            'confidence_score': int(confidence_score),
            'estimate_quality': quality,
            'last_transaction_date': last_date,
            'historical_median': float(historical_median),
            'ml_adjustment_factor': float(smoothed_prediction / historical_median) if historical_median > 0 else 1.0
        }
    
    def get_model_summary(self):
        """Return a summary of the model for documentation."""
        if not self.is_fitted:
            return "Model not fitted."
        
        annual_pct = (np.exp(self.annual_trend) - 1) * 100
        
        summary = f"""
ROBUST ML PRICE MODEL SUMMARY
=============================

Model Type: LightGBM Gradient Boosting + Linear Temporal Trend

TEMPORAL ADJUSTMENT
-------------------
- Estimated annual trend: {annual_pct:+.2f}%
- Prediction target date: {self.prediction_date.strftime('%Y-%m-%d')}

SPATIAL MODEL PERFORMANCE
-------------------------
- Cross-validation R²: {self.cv_r2:.3f}
- Cross-validation RMSE (log): {self.cv_rmse:.4f}

FEATURE IMPORTANCE
------------------
"""
        for _, row in self.feature_importance.iterrows():
            summary += f"- {row['feature']}: {row['importance']:.0f}\n"
        
        summary += f"""
HIERARCHICAL SMOOTHING
----------------------
- Min transactions for full weight: {MIN_TRANSACTIONS_FULL_WEIGHT}
- Zones with fewer transactions are smoothed toward regional average

GLOBAL STATISTICS
-----------------
- Global median price: €{self.global_median_price:,.0f}/m²
- Departments with priors: {len(self.department_medians)}
"""
        return summary
    
    def save(self, path):
        """Save model to disk."""
        joblib.dump({
            'spatial_model': self.spatial_model,
            'temporal_model': self.temporal_model,
            'annual_trend': self.annual_trend,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'global_median_price': self.global_median_price,
            'global_std_price': self.global_std_price,
            'department_medians': self.department_medians,
            'cv_rmse': self.cv_rmse,
            'cv_r2': self.cv_r2,
            'feature_importance': self.feature_importance,
            'prediction_date': self.prediction_date
        }, path)
        print(f"✓ Model saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load model from disk."""
        data = joblib.load(path)
        model = cls(prediction_date=data['prediction_date'])
        model.spatial_model = data['spatial_model']
        model.temporal_model = data['temporal_model']
        model.annual_trend = data['annual_trend']
        model.label_encoders = data['label_encoders']
        model.feature_columns = data['feature_columns']
        model.global_median_price = data['global_median_price']
        model.global_std_price = data['global_std_price']
        model.department_medians = data['department_medians']
        model.cv_rmse = data['cv_rmse']
        model.cv_r2 = data['cv_r2']
        model.feature_importance = data['feature_importance']
        model.is_fitted = True
        print(f"✓ Model loaded from {path}")
        return model
