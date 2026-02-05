#!/usr/bin/env python3
"""Divise les données par ranges de prix pour réseaux spécialisés"""

import pandas as pd
import json
from pathlib import Path

def split_by_ranges(input_file, output_prefix, ranges):
    """Divise un fichier CSV par ranges de prix"""
    data = pd.read_csv(input_file, header=None)
    price_col = data.columns[-1]

    print(f"\n{input_file}: {len(data)} échantillons")

    for range_name, min_price, max_price in ranges:
        mask = (data[price_col] >= min_price) & (data[price_col] < max_price)
        subset = data[mask]

        if len(subset) > 0:
            output_file = f"{output_prefix}_{range_name}.csv"
            subset.to_csv(output_file, index=False, header=False)
            print(f"  {range_name} [{min_price}-{max_price}): {len(subset)} → {output_file}")

    return data

def make_layer(size, activation="Tanh", l2=0.0, last=False):
    """Helper pour créer une couche"""
    layer = {
        "Size": size,
        "ActivatorType": activation,
        "Type": "Standard",
        "GradientAdjustmentParameters": {
            "Type": "FixedLearningRate" if last else "Momentum",
            "LearningRate": 0.01
        }
    }
    if not last:
        layer["GradientAdjustmentParameters"]["Momentum"] = 0.9
    if l2 > 0:
        layer["L2Regularization"] = l2
    return layer

def make_network(batch_size, hidden_layers):
    """Helper pour créer un réseau"""
    layers = [make_layer(s, l2=0.0001) for s in hidden_layers]
    layers.append(make_layer(1, "Identity", last=True))
    return {
        "InputSize": 7,
        "BatchSize": batch_size,
        "Initializer": "Xavier",
        "Layers": layers
    }

def create_network_configs(output_dir='pricing-data/experiments'):
    """Crée les configs pour les réseaux spécialisés"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    networks = [
        ('small', make_network(16, [50, 30, 15])),
        ('medium', make_network(32, [20, 10])),
        ('large', make_network(16, [10]))
    ]

    for range_name, network_config in networks:
        network_file = output_dir / f'network_{range_name}_prices.json'
        with open(network_file, 'w') as f:
            json.dump(network_config, f, indent=2)
        print(f"  Config réseau: {network_file}")

    # Créer les fichiers d'expériences
    for range_name, epochs in [('small', 10000), ('medium', 8000), ('large', 5000)]:
        expe_config = {
            "network description": f"pricing-data/experiments/network_{range_name}_prices.json",
            "training data": f"pricing-data/train_{range_name}.csv",
            "validation data": f"pricing-data/valid_{range_name}.csv",
            "epochs": epochs,
            "trained network": f"pricing-data/results/{range_name}_prices_learned.json",
            "cost function": "Quadratic",
            "initialize": True,
            "normalize": True,
            "normalizer file": f"pricing-data/results/{range_name}_prices_normalizer.json",
            "learning log file": f"pricing-data/results/{range_name}_prices_error.csv",
            "validation steps": 100,
            "final validation": f"pricing-data/results/{range_name}_prices_validation.csv",
            "gnuplot": False
        }

        expe_file = output_dir / f'expe_{range_name}_prices.json'
        with open(expe_file, 'w') as f:
            json.dump(expe_config, f, indent=2)
        print(f"  Config expérience: {expe_file}")

def main():
    print("\n=== Division des données par range de prix ===\n")

    ranges = [
        ('small', 0, 5),
        ('medium', 5, 15),
        ('large', 15, 100)
    ]

    split_by_ranges('pricing-data/train.csv', 'pricing-data/train', ranges)
    split_by_ranges('pricing-data/valid.csv', 'pricing-data/valid', ranges)

    print("\n=== Création des configurations ===\n")
    create_network_configs()

    print("\n=== Commandes d'entraînement ===\n")
    for name in ['small', 'medium', 'large']:
        print(f"mvn exec:java -Dexec.mainClass=\"fr.ensimag.deep.trainingConsole.Main\" \\")
        print(f"  -Dexec.args=\"-x pricing-data/experiments/expe_{name}_prices.json\"\n")

if __name__ == '__main__':
    main()
