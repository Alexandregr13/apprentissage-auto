# Visualisation HiPlot - Guide

## Installation

```bash
pip install hiplot
```

## Lancement

```bash
bash pricing-data/run_hiplot.sh
```

Ouvre automatiquement `pricing-data/results/hiplot_visualization.html` dans le navigateur.

## Utilisation

**Parallel Coordinates Plot:**
- Chaque ligne = une expérience
- Chaque axe = un hyperparamètre ou métrique
- Cliquer-glisser sur un axe pour filtrer

**Axes importants:**
- `validation_mse` : Performance (à minimiser)
- `normalized` : Avec/sans normalisation
- `learning_rate`, `momentum`, `l2_regularization` : Hyperparamètres
- `architecture` : Structure du réseau

**Actions:**
- Filtrer: Glisser sur un axe pour sélectionner une plage
- Comparer: Filtrer "normalized" pour voir l'impact de la normalisation
- Sélectionner: Cliquer sur une ligne pour la mettre en surbrillance

## Résultats clés

**Impact de la normalisation:** -97% d'erreur (MSE: 18 → 0.5)

Toutes les expériences normalisées atteignent MSE < 0.7, sauf `lr_high_norm` (LR trop élevé).
