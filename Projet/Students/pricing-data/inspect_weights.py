#!/usr/bin/env python3
import json
import numpy as np
import sys
import os

def inspect_network(network_file):
    exp_name = os.path.basename(network_file).replace('_learned.json', '')
    
    with open(network_file) as f:
        network = json.load(f)
    
    print(f"\n{exp_name}:")
    print(f"  Input: {network['InputSize']}, Layers: {len(network['Layers'])}")
    
    has_issues = False
    
    for i, layer in enumerate(network['Layers']):
        weights = np.array(layer['Weights']).flatten()
        biases = np.array(layer['Bias']) if 'Bias' in layer else np.array(layer.get('Biases', []))
        
        w_min, w_max = weights.min(), weights.max()
        w_mean, w_std = weights.mean(), weights.std()
        
        print(f"  Layer {i} ({layer['ActivatorType']}, size={layer['Size']}):")
        print(f"    Weights: [{w_min:.3f}, {w_max:.3f}], mean={w_mean:.3f}, std={w_std:.3f}")
        
        if len(biases) > 0:
            print(f"    Biases: [{biases.min():.3f}, {biases.max():.3f}], mean={biases.mean():.3f}")
        
        # Diagnostics
        issues = []
        if abs(weights.max()) > 100 or abs(weights.min()) > 100:
            issues.append(f"EXPLOSION (|max|={max(abs(w_max), abs(w_min)):.2f})")
            has_issues = True
        elif abs(weights.max()) > 10 or abs(weights.min()) > 10:
            issues.append(f"High weights (|max|={max(abs(w_max), abs(w_min)):.2f})")
        
        if abs(weights).mean() < 0.01:
            issues.append(f"Dead neurons? (|mean|={abs(weights).mean():.4f})")
            has_issues = True
        elif abs(weights).mean() < 0.1:
            issues.append(f"Low weights (|mean|={abs(weights).mean():.4f})")
        
        zeros = np.sum(np.abs(weights) < 0.001)
        if zeros > len(weights) * 0.5:
            issues.append(f"Many zeros ({zeros}/{len(weights)})")
            has_issues = True
        
        large_weights = np.sum(np.abs(weights) > 5)
        if large_weights > len(weights) * 0.1:
            issues.append(f"Saturation risk ({large_weights} weights > 5)")
        
        if issues:
            print(f"    Issues: {', '.join(issues)}")
        else:
            print(f"    Status: OK")
    
    return not has_issues

def inspect_all_experiments():
    # Detect if running from pricing-data/ or project root
    if os.path.exists('results'):
        results_dir = 'results'
    else:
        results_dir = 'pricing-data/results'
    
    experiments = [
        'baseline', 'baseline_norm', 'lr_high', 'lr_high_norm',
        'no_momentum', 'no_momentum_norm', 'deep', 'deep_norm',
        'relu', 'relu_norm', 'no_reg', 'no_reg_norm',
        'simple', 'simple_norm'
    ]
    
    print("Weight Inspection")
    print("=" * 70)
    
    results = []
    for exp in experiments:
        network_file = f'{results_dir}/{exp}_learned.json'
        if os.path.exists(network_file):
            is_healthy = inspect_network(network_file)
            results.append((exp, is_healthy))
        else:
            print(f'Missing: {network_file}')
    
    # Summary
    print("\n" + "=" * 70)
    if len(results) == 0:
        print("No networks found to analyze")
        return
    
    healthy = sum(1 for _, h in results if h)
    total = len(results)
    print(f"Healthy networks: {healthy}/{total} ({healthy/total*100:.1f}%)")
    
    print("\nHealthy:")
    for name, is_healthy in results:
        if is_healthy:
            print(f"  {name}")
    
    if healthy < total:
        print("\nWith issues:")
        for name, is_healthy in results:
            if not is_healthy:
                print(f"  {name}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        inspect_network(sys.argv[1])
    else:
        inspect_all_experiments()
