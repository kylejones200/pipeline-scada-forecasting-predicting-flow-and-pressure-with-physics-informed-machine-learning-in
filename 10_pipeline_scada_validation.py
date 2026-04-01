#!/usr/bin/env python3
"""
Validation script for Pipeline SCADA Forecasting blog code.
Tests all functions to ensure they run without errors.
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def generate_synthetic_scada_data(hours=672, seed=3363):
    """Generate realistic pipeline SCADA telemetry."""
    rng = np.random.default_rng(seed)
    
    time = pd.date_range("2024-01-01", periods=hours, freq="h")
    
    temp_celsius = (
        15 + 10 * np.sin(2 * np.pi * time.hour / 24) +
        rng.normal(0, 1, hours)
    )
    
    nominations = (
        200 + 40 * np.sin(2 * np.pi * np.arange(hours) / (7*24)) +
        rng.normal(0, 5, hours)
    )
    
    flow_mmscfd = (
        220 + 0.6 * nominations + -0.8 * temp_celsius +
        5 * np.sin(2 * np.pi * time.hour / 24) +
        rng.normal(0, 6, hours)
    )
    flow_mmscfd = np.clip(flow_mmscfd, 120, 350)
    
    k_drop = 0.05
    pressure_drop = (
        k_drop * (flow_mmscfd / 100) ** 1.8 +
        rng.normal(0, 0.8, hours)
    )
    
    p_suction_psig = (
        650 + 10 * np.sin(2 * np.pi * time.hour / 24) +
        rng.normal(0, 1.5, hours)
    )
    
    p_discharge_psig = p_suction_psig - pressure_drop
    
    return pd.DataFrame({
        'temp_celsius': temp_celsius,
        'nominations_mmscfd': nominations,
        'flow_mmscfd': flow_mmscfd,
        'p_suction_psig': p_suction_psig,
        'p_discharge_psig': p_discharge_psig
    }, index=time)

def create_lagged_features(series, lag_hours):
    """Create lagged features from time series."""
    lag_features = {}
    for lag in lag_hours:
        lag_features[f'{series.name}_lag{lag}h'] = series.shift(lag)
    return pd.DataFrame(lag_features)

def build_forecast_features(scada_data, lag_hours=[1, 2, 3, 6, 12, 24]):
    """Build comprehensive feature matrix for forecasting."""
    lagged_flow = create_lagged_features(scada_data['flow_mmscfd'], lag_hours)
    lagged_p_suction = create_lagged_features(scada_data['p_suction_psig'], lag_hours)
    lagged_p_discharge = create_lagged_features(scada_data['p_discharge_psig'], lag_hours)
    
    exogenous = scada_data[['temp_celsius', 'nominations_mmscfd']].copy()
    
    # Create day of week DataFrame with same index as scada_data
    day_of_week = pd.get_dummies(scada_data.index.dayofweek, prefix='dow', dtype=float)
    day_of_week.index = scada_data.index  # Ensure index matches
    
    feature_matrix = pd.concat([
        lagged_flow, lagged_p_suction, lagged_p_discharge,
        exogenous, day_of_week
    ], axis=1)
    
    return feature_matrix

def train_pipeline_forecast_models(dataset, test_hours=168):
    """Train separate ML models for flow and pressure forecasting."""
    train_data = dataset.iloc[:-test_hours]
    test_data = dataset.iloc[-test_hours:]
    
    feature_cols = [c for c in dataset.columns 
                   if c not in ['flow_mmscfd', 'p_suction_psig', 'p_discharge_psig']]
    
    X_train = train_data[feature_cols]
    X_test = test_data[feature_cols]
    
    y_train_flow = train_data['flow_mmscfd']
    y_test_flow = test_data['flow_mmscfd']
    
    y_train_p_suction = train_data['p_suction_psig']
    y_test_p_suction = test_data['p_suction_psig']
    
    y_train_p_discharge = train_data['p_discharge_psig']
    y_test_p_discharge = test_data['p_discharge_psig']
    
    kernel = ConstantKernel(1.0) * Matern(nu=1.5) + WhiteKernel()
    gp_flow = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-4, normalize_y=True, random_state=1,
        n_restarts_optimizer=5
    )
    
    gp_flow.fit(X_train, y_train_flow)
    pred_flow = gp_flow.predict(X_test)
    
    rf_pressure = RandomForestRegressor(
        n_estimators=300, max_depth=20, min_samples_split=10,
        random_state=1, n_jobs=-1
    )
    
    rf_pressure.fit(X_train, y_train_p_suction)
    pred_p_suction = rf_pressure.predict(X_test)
    
    rf_pressure_discharge = RandomForestRegressor(
        n_estimators=300, max_depth=20, min_samples_split=10,
        random_state=1, n_jobs=-1
    )
    rf_pressure_discharge.fit(X_train, y_train_p_discharge)
    pred_p_discharge = rf_pressure_discharge.predict(X_test)
    
    metrics = {
        'flow': {
            'mae': mean_absolute_error(y_test_flow, pred_flow),
            'rmse': np.sqrt(mean_squared_error(y_test_flow, pred_flow)),
            'mape': np.mean(np.abs((y_test_flow - pred_flow) / y_test_flow)) * 100
        },
        'p_suction': {
            'mae': mean_absolute_error(y_test_p_suction, pred_p_suction),
            'rmse': np.sqrt(mean_squared_error(y_test_p_suction, pred_p_suction)),
            'mape': np.mean(np.abs((y_test_p_suction - pred_p_suction) / y_test_p_suction)) * 100
        },
        'p_discharge': {
            'mae': mean_absolute_error(y_test_p_discharge, pred_p_discharge),
            'rmse': np.sqrt(mean_squared_error(y_test_p_discharge, pred_p_discharge)),
            'mape': np.mean(np.abs((y_test_p_discharge - pred_p_discharge) / y_test_p_discharge)) * 100
        }
    }
    
    return {
        'predictions': {
            'flow': pred_flow,
            'p_suction': pred_p_suction,
            'p_discharge': pred_p_discharge
        },
        'actuals': {
            'flow': y_test_flow,
            'p_suction': y_test_p_suction,
            'p_discharge': y_test_p_discharge
        },
        'metrics': metrics
    }

def enforce_physical_constraints(predictions, actuals):
    """Apply physics-based constraints to ML predictions."""
    pred_p_suction = predictions['p_suction']
    pred_p_discharge = predictions['p_discharge']
    
    MIN_SUCTION_PRESSURE = 550.0
    MIN_DISCHARGE_PRESSURE = 520.0
    MAX_PRESSURE_DROP = 40.0
    
    raw_violations = {
        'discharge_exceeds_suction': np.sum(pred_p_discharge > pred_p_suction),
        'dp_exceeds_max': np.sum((pred_p_suction - pred_p_discharge) > MAX_PRESSURE_DROP)
    }
    
    pred_p_suction_constrained = np.maximum(pred_p_suction, MIN_SUCTION_PRESSURE)
    pred_p_discharge_constrained = np.maximum(pred_p_discharge, MIN_DISCHARGE_PRESSURE)
    
    pred_p_discharge_constrained = np.minimum(
        pred_p_discharge_constrained,
        pred_p_suction_constrained - 1.0
    )
    
    pred_p_discharge_constrained = np.maximum(
        pred_p_discharge_constrained,
        pred_p_suction_constrained - MAX_PRESSURE_DROP
    )
    
    post_violations = {
        'discharge_exceeds_suction': np.sum(pred_p_discharge_constrained > pred_p_suction_constrained)
    }
    
    original_mae = mean_absolute_error(actuals['p_discharge'], pred_p_discharge)
    constrained_mae = mean_absolute_error(actuals['p_discharge'], pred_p_discharge_constrained)
    
    return {
        'raw_violations': raw_violations,
        'post_violations': post_violations,
        'accuracy_impact': constrained_mae - original_mae
    }

def main():
    """Run validation tests."""
    print("=" * 70)
    print("PIPELINE SCADA FORECASTING - CODE VALIDATION")
    print("=" * 70)
    
    np.random.seed(3363)
    
    print("\n1. Testing SCADA data generation...")
    scada = generate_synthetic_scada_data(hours=24*60)  # 60 days for adequate training data
    print(f"   ✓ Generated {len(scada)} hours of data")
    print(f"   ✓ Flow range: {scada['flow_mmscfd'].min():.1f} to {scada['flow_mmscfd'].max():.1f} MMscf/d")
    print(f"   ✓ Avg pressure drop: {(scada['p_suction_psig'] - scada['p_discharge_psig']).mean():.1f} psig")
    
    print("\n2. Testing feature engineering...")
    features = build_forecast_features(scada, lag_hours=[1, 2, 3, 6, 12, 24])
    print(f"   - Features shape before concat: {features.shape}")
    print(f"   - NaN count in features: {features.isna().sum().sum()}")
    targets = scada[['flow_mmscfd', 'p_suction_psig', 'p_discharge_psig']]
    print(f"   - Targets shape: {targets.shape}")
    dataset = pd.concat([features, targets], axis=1)
    print(f"   - Dataset shape before dropna: {dataset.shape}")
    dataset = dataset.dropna()
    print(f"   ✓ Features created: {features.shape[1]}")
    print(f"   ✓ Dataset size after dropna: {len(dataset)}")
    print(f"   ✓ Feature categories: lags, exogenous, temporal")
    
    print("\n3. Testing model training...")
    results = train_pipeline_forecast_models(dataset, test_hours=24*7)
    print(f"   ✓ Flow MAE: {results['metrics']['flow']['mae']:.2f} MMscf/d")
    print(f"   ✓ Suction MAE: {results['metrics']['p_suction']['mae']:.2f} psig")
    print(f"   ✓ Discharge MAE: {results['metrics']['p_discharge']['mae']:.2f} psig")
    
    print("\n4. Testing constraint enforcement...")
    constrained = enforce_physical_constraints(results['predictions'], results['actuals'])
    print(f"   ✓ Raw violations: {constrained['raw_violations']['discharge_exceeds_suction']}")
    print(f"   ✓ Post-constraint violations: {constrained['post_violations']['discharge_exceeds_suction']}")
    print(f"   ✓ Accuracy impact: {constrained['accuracy_impact']:.2f} psig")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)

if __name__ == "__main__":
    main()

