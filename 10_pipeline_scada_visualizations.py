#!/usr/bin/env python3
import sys
import os

# Add parent directory to path to import plot_style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import set_tufte_defaults, apply_tufte_style, save_tufte_figure, COLORS

"""
Generate visualizations for Pipeline SCADA Forecasting blog post.
Uses minimalist styling with serif fonts, clean axes, and high-quality output.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import sys
import os

# Add parent directory to path to import plot_style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import set_tufte_defaults, apply_tufte_style, save_tufte_figure, COLORS

# Import Tufte plotting utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tda_utils import setup_tufte_plot, TufteColors



def save_fig(filename):
    """Save plot in the standard minimalist format."""
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def generate_scada_data(hours=1440, seed=3363):
    """Generate realistic pipeline SCADA telemetry."""
    rng = np.random.default_rng(seed)
    time = pd.date_range("2024-01-01", periods=hours, freq="h")
    
    temp = 15 + 10 * np.sin(2 * np.pi * time.hour / 24) + rng.normal(0, 1, hours)
    noms = 200 + 40 * np.sin(2 * np.pi * np.arange(hours) / (7*24)) + rng.normal(0, 5, hours)
    
    flow = 220 + 0.6 * noms + -0.8 * temp + 5 * np.sin(2 * np.pi * time.hour / 24) + rng.normal(0, 6, hours)
    flow = np.clip(flow, 120, 350)
    
    dp = 0.05 * (flow / 100) ** 1.8 + rng.normal(0, 0.8, hours)
    p_suc = 650 + 10 * np.sin(2 * np.pi * time.hour / 24) + rng.normal(0, 1.5, hours)
    p_dis = p_suc - dp
    
    return pd.DataFrame({
        'temp': temp, 'nominations': noms, 'flow': flow,
        'p_suction': p_suc, 'p_discharge': p_dis
    }, index=time)

def make_lags(s, lags):
    """Create lagged features."""
    out = {}
    for L in lags:
        out[f"{s.name}_lag{L}"] = s.shift(L)
    return pd.DataFrame(out)

def train_models(df):
    """Train simplified models for visualization."""
    lags = [1, 2, 3, 6, 12, 24]
    
    dow = pd.get_dummies(df.index.dayofweek, prefix='dow', dtype=float)
    dow.index = df.index
    
    X = pd.concat([
        make_lags(df['flow'], lags),
        make_lags(df['p_suction'], lags),
        make_lags(df['p_discharge'], lags),
        df[['temp', 'nominations']],
        dow
    ], axis=1)
    
    data = pd.concat([X, df[['flow', 'p_suction', 'p_discharge']]], axis=1).dropna()
    
    train = data.iloc[:-24*7]
    test = data.iloc[-24*7:]
    
    Xtr = train.drop(columns=['flow', 'p_suction', 'p_discharge'])
    Xte = test.drop(columns=['flow', 'p_suction', 'p_discharge'])
    
    # Train flow model
    kernel = ConstantKernel(1.0) * Matern(nu=1.5) + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True, random_state=1)
    gp.fit(Xtr, train['flow'])
    pred_f = gp.predict(Xte)
    
    # Train pressure models
    rf = RandomForestRegressor(n_estimators=200, random_state=1, n_jobs=-1)
    rf.fit(Xtr, train['p_suction'])
    pred_ps = rf.predict(Xte)
    
    rf2 = RandomForestRegressor(n_estimators=200, random_state=1, n_jobs=-1)
    rf2.fit(Xtr, train['p_discharge'])
    pred_pd = rf2.predict(Xte)
    
    # Apply constraints
    pred_ps = np.maximum(pred_ps, 550)
    pred_pd = np.maximum(pred_pd, 520)
    pred_pd = np.minimum(pred_pd, pred_ps - 1)
    
    return test.index, test['flow'].values, pred_f, test['p_suction'].values, pred_ps, test['p_discharge'].values, pred_pd

def create_main_visualization():
    """Create main SCADA forecasting visualization."""
    np.random.seed(3363)
    df = generate_scada_data(hours=24*60)
    tidx, yf, pf, yps, pps, ypd, ppd = train_models(df)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    
    # Flow panel
    ax1.plot(tidx, yf, color='black', linewidth=1.5, label='Actual')
    ax1.plot(tidx, pf, color='gray', linewidth=1.5, linestyle='--', label='Forecast')
    
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_position(("outward", 5))
    ax1.spines["bottom"].set_position(("outward", 5))
    ax1.set_title('Flow Rate Forecast', fontsize=12, fontweight="bold", loc="left")
    ax1.set_ylabel('Flow (MMscf/d)', fontsize=10)
    ax1.legend(loc='upper right', frameon=False, fontsize=9)
    
    # Suction pressure panel
    ax2.plot(tidx, yps, color='black', linewidth=1.5, label='Actual')
    ax2.plot(tidx, pps, color='gray', linewidth=1.5, linestyle='--', label='Forecast')
    
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_position(("outward", 5))
    ax2.spines["bottom"].set_position(("outward", 5))
    ax2.set_title('Suction Pressure Forecast', fontsize=12, fontweight="bold", loc="left")
    ax2.set_ylabel('Pressure (psig)', fontsize=10)
    ax2.legend(loc='upper right', frameon=False, fontsize=9)
    
    # Discharge pressure panel
    ax3.plot(tidx, ypd, color='black', linewidth=1.5, label='Actual')
    ax3.plot(tidx, ppd, color='gray', linewidth=1.5, linestyle='--', label='Forecast')
    
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_position(("outward", 5))
    ax3.spines["bottom"].set_position(("outward", 5))
    ax3.set_title('Discharge Pressure Forecast', fontsize=12, fontweight="bold", loc="left")
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('Pressure (psig)', fontsize=10)
    ax3.legend(loc='upper right', frameon=False, fontsize=9)
    
    save_fig('10_pipeline_scada_main.png')
    print("✓ Created: 10_pipeline_scada_main.png")

def create_accuracy_visualization():
    """Create forecast accuracy visualization."""
    np.random.seed(3363)
    df = generate_scada_data(hours=24*60)
    tidx, yf, pf, yps, pps, ypd, ppd = train_models(df)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Scatter: Actual vs Predicted
    ax1.scatter(yf, pf, s=30, alpha=0.6, color='white', edgecolors='black', linewidths=0.8, label='Flow')
    ax1.scatter(yps, pps, s=30, alpha=0.6, color='gray', edgecolors='black', linewidths=0.8, marker='s', label='Suction P')
    
    # Perfect prediction line
    all_vals = np.concatenate([yf, yps])
    min_val, max_val = all_vals.min(), all_vals.max()
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Perfect')
    
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_position(("outward", 5))
    ax1.spines["bottom"].set_position(("outward", 5))
    ax1.set_title('Forecast Accuracy', fontsize=12, fontweight="bold", loc="left")
    ax1.set_xlabel('Actual', fontsize=10)
    ax1.set_ylabel('Predicted', fontsize=10)
    ax1.legend(loc='upper left', frameon=False, fontsize=9)
    ax1.set_aspect('equal')
    
    # Residual distribution
    res_f = yf - pf
    res_ps = yps - pps
    
    bins = np.linspace(-15, 15, 25)
    ax2.hist(res_f, bins=bins, alpha=0.7, color='white', edgecolor='black', linewidth=1.5, label='Flow')
    ax2.hist(res_ps, bins=bins, alpha=0.7, color='gray', edgecolor='black', linewidth=1.5, label='Pressure')
    
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_position(("outward", 5))
    ax2.spines["bottom"].set_position(("outward", 5))
    mae_f = mean_absolute_error(yf, pf)
    mae_ps = mean_absolute_error(yps, pps)
    
    stats_text = f'Flow MAE: {mae_f:.1f}\nPressure MAE: {mae_ps:.1f}'
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1))
    
    ax2.set_title('Forecast Residuals', fontsize=12, fontweight="bold", loc="left")
    ax2.set_xlabel('Residual (Actual - Predicted)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.legend(loc='upper left', frameon=False, fontsize=9)
    
    save_fig('10_pipeline_scada_accuracy.png')
    print("✓ Created: 10_pipeline_scada_accuracy.png")

def main():
    """Generate all visualizations."""
    set_tufte_defaults()
    print("=" * 60)
    print("PIPELINE SCADA FORECASTING - VISUALIZATION GENERATION")
    print("=" * 60)
    print()
    
    plt.rcParams['font.family'] = 'serif'
    
    print("Creating visualizations...")
    create_main_visualization()
    create_accuracy_visualization()
    
    print()
    print("=" * 60)
    print("All visualizations created successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

