#!/usr/bin/env python3
"""Visualisation interactive HiPlot pour analyser les expériences de pricing"""

import json
from pathlib import Path
import pandas as pd

try:
    import hiplot as hip
except ImportError:
    print("HiPlot non installé. Exécutez: pip install hiplot")
    exit(1)


def parse_network_config(network_file):
    """Extrait les hyperparamètres du fichier réseau"""
    with open(network_file) as f:
        config = json.load(f)

    layer_sizes = [config['InputSize']] + [layer['Size'] for layer in config['Layers']]
    architecture = '-'.join(map(str, layer_sizes))
    first_layer = config['Layers'][0]

    return {
        'architecture': architecture,
        'hidden_layers': len(config['Layers']) - 1,
        'learning_rate': first_layer['GradientAdjustmentParameters']['LearningRate'],
        'momentum': first_layer['GradientAdjustmentParameters'].get('Momentum', 0.0),
        'l2_regularization': first_layer.get('L2Regularization', 0.0),
        'activation': first_layer['ActivatorType'],
    }


def get_final_metrics(error_file):
    """Extrait les métriques finales du fichier d'erreurs"""
    try:
        df = pd.read_csv(error_file, header=None, sep=' ')
        final_train_mse = df[0].iloc[-1]
        final_val_mse = df[1].iloc[-1]
        final_rmse = df[2].iloc[-1] if len(df.columns) > 2 else (final_val_mse ** 0.5)
        generalization_gap = ((final_val_mse - final_train_mse) / final_train_mse) * 100

        return {
            'train_mse': final_train_mse,
            'validation_mse': final_val_mse,
            'rmse': final_rmse,
            'generalization_gap_%': generalization_gap
        }
    except:
        return None


def collect_experiments():
    """Collecte toutes les données des expériences"""
    experiments_dir = Path('pricing-data/experiments')
    experiment_files = sorted(experiments_dir.glob('expe_*.json'))
    experiments_data = []

    for expe_file in experiment_files:
        exp_name = expe_file.stem.replace('expe_', '')

        with open(expe_file) as f:
            expe_config = json.load(f)

        network_file = Path(expe_config['network description'])
        if not network_file.exists():
            continue

        network_params = parse_network_config(network_file)
        error_file = Path(expe_config['learning log file'])
        metrics = get_final_metrics(error_file) if error_file.exists() else None
        normalized = expe_config.get('normalize', False) or '_norm' in exp_name

        experiment_data = {
            'experiment_name': exp_name,
            'normalized': 'Yes' if normalized else 'No',
            **network_params
        }

        if metrics:
            experiment_data.update(metrics)

        experiments_data.append(experiment_data)

    return experiments_data


def main():
    print("=" * 60)
    print("VISUALISATION HIPLOT - EXPÉRIENCES DE PRICING")
    print("=" * 60)

    experiments = collect_experiments()
    print(f"\n{len(experiments)} expériences collectées\n")

    df = pd.DataFrame(experiments).sort_values('validation_mse', na_position='last')

    print("Top 5 meilleures expériences:")
    print(df[['experiment_name', 'normalized', 'architecture', 'validation_mse']].head().to_string(index=False))

    # Sauvegarder CSV
    output_csv = 'pricing-data/results/hiplot_data.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nDonnées: {output_csv}")

    # Créer visualisation HiPlot
    hip_exp = hip.Experiment.from_iterable(experiments)
    html_output = 'pricing-data/results/hiplot_visualization.html'
    hip_exp.to_html(html_output)
    print(f"Visualisation: {html_output}")

    # Ouvrir dans navigateur
    import webbrowser, os
    webbrowser.open('file://' + os.path.abspath(html_output))
    print("\nOK: Visualisation ouverte dans le navigateur")


if __name__ == '__main__':
    main()
