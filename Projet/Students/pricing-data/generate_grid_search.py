#!/usr/bin/env python3
"""Génère un grid search intelligent d'hyperparamètres"""

import json
import itertools
from pathlib import Path

def make_layer(size, activation, l2, momentum, lr, last=False):
    """Crée une couche"""
    layer = {
        "Size": size,
        "ActivatorType": activation,
        "Type": "Standard",
        "GradientAdjustmentParameters": {
            "Type": "FixedLearningRate" if last else "Momentum",
            "LearningRate": lr
        }
    }
    if not last:
        layer["GradientAdjustmentParameters"]["Momentum"] = momentum
    if l2 > 0 and not last:
        layer["L2Regularization"] = l2
    return layer

def make_network(architecture, activation, batch_size, lr, momentum, l2):
    """Crée un réseau avec les hyperparamètres donnés"""
    layers = [make_layer(s, activation, l2, momentum, lr) for s in architecture]
    layers.append(make_layer(1, "Identity", 0.0, momentum, lr, last=True))

    return {
        "InputSize": 7,
        "BatchSize": batch_size,
        "Initializer": "Xavier",
        "Layers": layers
    }

def is_valid_config(lr, momentum, architecture, l2):
    """Filtre les configs absurdes"""
    # LR trop élevé + réseau profond = instable
    if lr >= 0.02 and len(architecture) > 2:
        return False

    # Réseau profond sans régularisation = risque overfitting
    if len(architecture) >= 3 and l2 == 0.0:
        return False

    # Momentum 0 + LR faible = trop lent
    if momentum == 0.0 and lr < 0.01:
        return False

    return True

def generate_grid_search():
    """Génère un grid search intelligent"""

    # Paramètres à explorer (~50 configs)
    learning_rates = [0.01]  # Garde le meilleur
    momentums = [0.9]        # Garde le meilleur
    architectures = [
        ([7], 'tiny'),
        ([10], 'simple'),
        ([15], 'simple15'),
        ([20, 10], 'standard'),
        ([30, 15], 'medium'),
        ([30, 20, 10], 'deep'),
        ([50, 30, 15], 'very_deep')
    ]
    activations = ['Tanh', 'Relu']  # IMPORTANT: 'Relu' pas 'ReLU' !
    l2_regs = [0.0001, 0.001]
    batch_sizes = [16, 32]

    configs = []

    for lr, momentum, (arch, arch_name), activation, l2, batch_size in itertools.product(
        learning_rates, momentums, architectures, activations, l2_regs, batch_sizes
    ):
        # Filtrer les configs invalides
        if not is_valid_config(lr, momentum, arch, l2):
            continue

        # Créer un nom descriptif
        name = f"grid_{arch_name}_lr{lr}_m{momentum}_l2{l2}_{activation.lower()}_bs{batch_size}"

        config = {
            'name': name,
            'learning_rate': lr,
            'momentum': momentum,
            'architecture': arch,
            'architecture_name': arch_name,
            'activation': activation,
            'l2_regularization': l2,
            'batch_size': batch_size
        }

        configs.append(config)

    return configs

def create_experiment_files(configs, output_dir='pricing-data/experiments'):
    """Crée les fichiers JSON pour chaque config"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for i, config in enumerate(configs, 1):
        name = config['name']

        # Créer le fichier réseau
        network = make_network(
            architecture=config['architecture'],
            activation=config['activation'],
            batch_size=config['batch_size'],
            lr=config['learning_rate'],
            momentum=config['momentum'],
            l2=config['l2_regularization']
        )

        network_file = output_dir / f'network_{name}.json'
        with open(network_file, 'w') as f:
            json.dump(network, f, indent=2)

        # Créer le fichier expérience
        expe = {
            "network description": f"pricing-data/experiments/network_{name}.json",
            "training data": "pricing-data/train.csv",
            "validation data": "pricing-data/valid.csv",
            "epochs": 5000,
            "trained network": f"pricing-data/results/{name}_learned.json",
            "cost function": "Quadratic",
            "initialize": True,
            "normalize": True,
            "normalizer file": f"pricing-data/results/{name}_normalizer.json",
            "learning log file": f"pricing-data/results/{name}_error.csv",
            "validation steps": 100,
            "final validation": f"pricing-data/results/{name}_validation.csv",
            "gnuplot": False
        }

        expe_file = output_dir / f'expe_{name}.json'
        with open(expe_file, 'w') as f:
            json.dump(expe, f, indent=2)

    return len(configs)

def generate_run_script(configs, output_file='pricing-data/run_grid_search.sh'):
    """Génère un script bash pour lancer tous les entraînements"""
    with open(output_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Grid search automatique\n\n")

        for i, config in enumerate(configs, 1):
            name = config['name']
            f.write(f"echo '=== [{i}/{len(configs)}] {name} ==='\n")
            f.write(f"mvn exec:java -Dexec.mainClass=\"fr.ensimag.deep.trainingConsole.Main\" \\\n")
            f.write(f"  -Dexec.args=\"-x pricing-data/experiments/expe_{name}.json\"\n\n")

    # Rendre le script exécutable
    import os
    os.chmod(output_file, 0o755)

def main():
    print("\n=== Génération du Grid Search ===\n")

    configs = generate_grid_search()
    print(f"Configurations générées : {len(configs)}")

    # Afficher quelques stats
    lrs = set(c['learning_rate'] for c in configs)
    archs = set(c['architecture_name'] for c in configs)

    print(f"\nLearning rates : {sorted(lrs)}")
    print(f"Architectures : {sorted(archs)}")

    # Estimer le temps
    avg_time_per_network = 25  # secondes
    total_time = len(configs) * avg_time_per_network
    hours = total_time / 3600

    print(f"\nTemps estimé : {total_time}s = {hours:.1f}h")

    # Demander confirmation
    response = input(f"\nCréer les {len(configs)} fichiers ? (y/n) : ")
    if response.lower() != 'y':
        print("Annulé")
        return

    # Créer les fichiers
    n_created = create_experiment_files(configs)
    print(f"\n✅ {n_created * 2} fichiers créés (network + expe)")

    # Créer le script bash
    generate_run_script(configs)
    print(f"✅ Script d'exécution : pricing-data/run_grid_search.sh")

    print(f"\nLancer avec : bash pricing-data/run_grid_search.sh")

if __name__ == '__main__':
    main()
