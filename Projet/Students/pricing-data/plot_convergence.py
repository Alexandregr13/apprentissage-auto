#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_convergence(error_file, output_file=None):
    exp_name = os.path.basename(error_file).replace('_error.csv', '')
    df = pd.read_csv(error_file, header=None, sep=' ', names=['MSE', 'RMSE'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = np.arange(len(df)) * 100
    
    # MSE
    axes[0].plot(epochs, df['MSE'], linewidth=2)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('MSE')
    axes[0].set_title(f'{exp_name} - MSE')
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.95, 0.95, f'Final: {df["MSE"].iloc[-1]:.2f}',
                 transform=axes[0].transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # RMSE
    axes[1].plot(epochs, df['RMSE'], linewidth=2)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title(f'{exp_name} - RMSE')
    axes[1].grid(True, alpha=0.3)
    axes[1].text(0.95, 0.95, f'Final: {df["RMSE"].iloc[-1]:.2f}',
                 transform=axes[1].transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Convergence: {exp_name}')
    plt.tight_layout()
    
    if output_file is None:
        output_file = error_file.replace('_error.csv', '_convergence.png')
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_file}')
    
    # Stats
    initial_mse = df["MSE"].iloc[0]
    final_mse = df["MSE"].iloc[-1]
    reduction = (initial_mse - final_mse) / initial_mse * 100
    last_10_pct = int(len(df) * 0.1)
    variation = df['MSE'].iloc[-last_10_pct:].std()
    
    print(f'{exp_name}: MSE {initial_mse:.2f} -> {final_mse:.2f} (-{reduction:.1f}%), std={variation:.4f}')
    
    plt.close()

def plot_all_experiments():
    results_dir = 'pricing-data/results'
    experiments = [
        'baseline', 'baseline_norm', 'lr_high', 'lr_high_norm',
        'no_momentum', 'no_momentum_norm', 'deep', 'deep_norm',
        'relu', 'relu_norm', 'no_reg', 'no_reg_norm',
        'simple', 'simple_norm'
    ]
    
    print(f"Processing {len(experiments)} experiments...")
    
    for exp in experiments:
        error_file = f'{results_dir}/{exp}_error.csv'
        if os.path.exists(error_file):
            plot_convergence(error_file)
        else:
            print(f'Missing: {error_file}')
    
    print(f"\nGraphs saved in: {results_dir}/*_convergence.png")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        error_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        plot_convergence(error_file, output_file)
    else:
        plot_all_experiments()
