#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
import os

def generate_complete_analysis():
    results_dir = 'pricing-data/results'
    experiments = {
        'baseline': 'Baseline (Sans norm)', 'baseline_norm': 'Baseline (Avec norm)',
        'lr_high': 'LR High (Sans norm)', 'lr_high_norm': 'LR High (Avec norm)',
        'no_momentum': 'No Momentum (Sans norm)', 'no_momentum_norm': 'No Momentum (Avec norm)',
        'deep': 'Deep (Sans norm)', 'deep_norm': 'Deep (Avec norm)',
        'relu': 'ReLU (Sans norm)', 'relu_norm': 'ReLU (Avec norm)',
        'no_reg': 'No Reg (Sans norm)', 'no_reg_norm': 'No Reg (Avec norm)',
        'simple': 'Simple (Sans norm)', 'simple_norm': 'Simple (Avec norm)'
    }
    
    data = []
    print("Complete Analysis")
    print("=" * 80)
    
    for exp, name in experiments.items():
        error_file = f'{results_dir}/{exp}_error.csv'
        network_file = f'{results_dir}/{exp}_learned.json'
        
        if not os.path.exists(error_file):
            continue
        
        # Charger l'erreur
        df_error = pd.read_csv(error_file, header=None, sep=' ', names=['MSE', 'RMSE'])
        final_mse = df_error['MSE'].iloc[-1]
        final_rmse = df_error['RMSE'].iloc[-1]
        initial_mse = df_error['MSE'].iloc[0]
        reduction = (initial_mse - final_mse) / initial_mse * 100
        
        # Tendance et stabilité
        last_20_pct = int(len(df_error) * 0.2)
        recent_mse = df_error['MSE'].iloc[-last_20_pct:]
        trend = np.polyfit(range(len(recent_mse)), recent_mse, 1)[0]
        variation = recent_mse.std() / recent_mse.mean()
        
        # Overfitting
        overfitting = trend > 0.01
        
        # Convergence
        last_10_pct = int(len(df_error) * 0.1)
        converged = df_error['MSE'].iloc[-last_10_pct:].std() < 0.1
        
        # Poids
        weight_status = "N/A"
        if os.path.exists(network_file):
            try:
                with open(network_file) as f:
                    network = json.load(f)
                
                all_weights = []
                for layer in network['Layers']:
                    weights = np.array(layer['Weights']).flatten()
                    all_weights.extend(weights)
                
                all_weights = np.array(all_weights)
                max_abs = max(abs(all_weights.max()), abs(all_weights.min()))
                
                if max_abs > 100:
                    weight_status = "Explosion"
                elif max_abs > 10:
                    weight_status = "High"
                elif abs(all_weights).mean() < 0.01:
                    weight_status = "Dead"
                else:
                    weight_status = "OK"
            except:
                weight_status = "Error"
        
        # Statut global
        if final_mse < 1.0 and not overfitting and converged and weight_status == "OK":
            status = "Excellent"
        elif final_mse < 5.0 and not overfitting:
            status = "Good"
        elif final_mse < 15.0:
            status = "Medium"
        else:
            status = "Poor"
        
        data.append({
            'Experiment': name,
            'MSE': final_mse,
            'RMSE': final_rmse,
            'Reduction': reduction,
            'Converged': 'Yes' if converged else 'No',
            'Overfitting': 'Yes' if overfitting else 'No',
            'Stable': 'Yes' if variation < 0.05 else 'No',
            'Weights': weight_status,
            'Status': status
        })
    
    df = pd.DataFrame(data)
    csv_file = f'{results_dir}/complete_analysis.csv'
    df.to_csv(csv_file, index=False)
    print(f"\nSaved: {csv_file}")
    
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    
    print("\n" + "=" * 80)
    excellent = len(df[df['Status'] == 'Excellent'])
    good = len(df[df['Status'] == 'Good'])
    medium = len(df[df['Status'] == 'Medium'])
    poor = len(df[df['Status'] == 'Poor'])
    
    print(f"\nPerformance: Excellent={excellent}, Good={good}, Medium={medium}, Poor={poor}")
    print(f"Overfitting: {len(df[df['Overfitting'] == 'Yes'])}/{len(df)}")
    print(f"Converged: {len(df[df['Converged'] == 'Yes'])}/{len(df)}")
    print(f"Stable: {len(df[df['Stable'] == 'Yes'])}/{len(df)}")
    
    print("\n" + "=" * 80)
    print("TOP 3")
    print("=" * 80)
    
    top3 = df.nsmallest(3, 'MSE')
    for i, (idx, row) in enumerate(top3.iterrows(), 1):
        print(f"\n{i}. {row['Experiment']}: MSE={row['MSE']:.2f}, Status={row['Status']}")
        print(f"   Converged={row['Converged']}, Overfitting={row['Overfitting']}, Weights={row['Weights']}")
    
    best = df.nsmallest(1, 'MSE').iloc[0]
    print("\n" + "=" * 80)
    print(f"Best: {best['Experiment']} (MSE={best['MSE']:.2f}, {best['Status']})")
    print(f"File: {results_dir}/{best['Experiment'].split()[0].lower()}_learned.json")
    print("=" * 80)

if __name__ == '__main__':
    generate_complete_analysis()
