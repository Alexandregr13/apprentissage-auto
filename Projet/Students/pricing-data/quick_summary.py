#!/usr/bin/env python3
"""Résumé rapide des résultats du grid search"""

import json
import pandas as pd
from pathlib import Path

def get_results():
    """Collecte les résultats de toutes les expériences"""
    experiments_dir = Path('pricing-data/experiments')
    results_dir = Path('pricing-data/results')

    data = []

    for expe_file in sorted(experiments_dir.glob('expe_*.json')):
        exp_name = expe_file.stem.replace('expe_', '')

        # Charger l'expérience
        with open(expe_file) as f:
            expe = json.load(f)

        # Charger le réseau
        network_file = Path(expe['network description'])
        if not network_file.exists():
            continue

        with open(network_file) as f:
            network = json.load(f)

        # Extraire les hyperparamètres
        first_layer = network['Layers'][0]
        layer_sizes = [network['InputSize']] + [l['Size'] for l in network['Layers']]
        architecture = '-'.join(map(str, layer_sizes))

        # Charger les résultats
        error_file = results_dir / expe_file.stem.replace('expe_', '') + '_error.csv'

        if not error_file.exists():
            continue

        try:
            df = pd.read_csv(error_file, header=None, sep=' ')
            train_mse = df[0].iloc[-1]
            val_mse = df[1].iloc[-1]
            rmse = (val_mse ** 0.5)
        except:
            continue

        data.append({
            'experiment': exp_name,
            'architecture': architecture,
            'batch_size': network.get('BatchSize', 32),
            'activation': first_layer['ActivatorType'],
            'learning_rate': first_layer['GradientAdjustmentParameters']['LearningRate'],
            'momentum': first_layer['GradientAdjustmentParameters'].get('Momentum', 0.0),
            'l2_reg': first_layer.get('L2Regularization', 0.0),
            'train_mse': train_mse,
            'val_mse': val_mse,
            'rmse': rmse,
            'normalized': expe.get('normalize', False) or '_norm' in exp_name
        })

    return pd.DataFrame(data)

def main():
    print("\n" + "="*80)
    print("RÉSUMÉ DES RÉSULTATS")
    print("="*80 + "\n")

    df = get_results()

    if len(df) == 0:
        print("Aucun résultat trouvé. Lancer d'abord les entraînements.")
        return

    print(f"Total expériences : {len(df)}\n")

    # Top 10 par validation MSE
    print("="*80)
    print("TOP 10 - Meilleure Validation MSE")
    print("="*80)

    top10 = df.nsmallest(10, 'val_mse')[['experiment', 'architecture', 'activation',
                                           'batch_size', 'l2_reg', 'val_mse', 'rmse']]

    print(top10.to_string(index=False))

    # Stats par architecture
    print("\n" + "="*80)
    print("STATISTIQUES PAR ARCHITECTURE")
    print("="*80)

    arch_stats = df.groupby('architecture')['val_mse'].agg(['mean', 'min', 'max', 'count'])
    arch_stats = arch_stats.sort_values('mean')
    print(arch_stats.to_string())

    # Stats par activation
    print("\n" + "="*80)
    print("STATISTIQUES PAR ACTIVATION")
    print("="*80)

    act_stats = df.groupby('activation')['val_mse'].agg(['mean', 'min', 'count'])
    act_stats = act_stats.sort_values('mean')
    print(act_stats.to_string())

    # Stats par batch size
    print("\n" + "="*80)
    print("STATISTIQUES PAR BATCH SIZE")
    print("="*80)

    bs_stats = df.groupby('batch_size')['val_mse'].agg(['mean', 'min', 'count'])
    bs_stats = bs_stats.sort_values('mean')
    print(bs_stats.to_string())

    # Stats par L2 reg
    print("\n" + "="*80)
    print("STATISTIQUES PAR L2 REGULARIZATION")
    print("="*80)

    l2_stats = df.groupby('l2_reg')['val_mse'].agg(['mean', 'min', 'count'])
    l2_stats = l2_stats.sort_values('mean')
    print(l2_stats.to_string())

    # Meilleur réseau
    print("\n" + "="*80)
    print("🏆 MEILLEUR RÉSEAU")
    print("="*80)

    best = df.loc[df['val_mse'].idxmin()]
    print(f"\nExpérience : {best['experiment']}")
    print(f"Architecture : {best['architecture']}")
    print(f"Activation : {best['activation']}")
    print(f"Batch Size : {int(best['batch_size'])}")
    print(f"L2 Reg : {best['l2_reg']}")
    print(f"Validation MSE : {best['val_mse']:.4f}")
    print(f"RMSE : {best['rmse']:.4f}")

    # Sauvegarder CSV complet
    output_file = 'pricing-data/results/all_experiments_summary.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Résumé complet sauvegardé : {output_file}")

if __name__ == '__main__':
    main()
