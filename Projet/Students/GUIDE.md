# Projet Deep Learning - Pricing Neural Network

## Vue d'ensemble

Ce projet implémente un réseau de neurones pour prédire les prix dans un problème de pricing.
Deux phases d'expérimentation ont été réalisées :
1. **Phase 1 (14 expériences)** : Analyse comparative baseline avec/sans normalisation
2. **Phase 2 (68 expériences)** : Grid search systématique pour optimisation des hyperparamètres

## Installation

```bash
mvn clean compile
```

## Lancer une Expérience

```bash
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_baseline_norm.json"
```

## Phase 2 : Grid Search (68 expériences)

```bash
# Générer les configs
python3 pricing-data/generate_grid_search.py

# Lancer les entraînements
bash pricing-data/run_grid_search.sh
```

## Analyse des Résultats

### Résumé rapide
```bash
python3 pricing-data/quick_summary.py
```

### HiPlot (visualisation interactive)
```bash
python3 pricing-data/hiplot_analysis.py
```

### Analyse par prix
```bash
python3 pricing-data/analyze_error_by_price.py
```

## Meilleur Réseau (68 expériences)

**grid_simple15_lr0.01_m0.9_l20.001_tanh_bs32**
- Architecture : 7-15-1 (1 couche cachée, 15 neurones)
- Activation : Tanh
- LR : 0.01, Momentum : 0.9
- L2 : 0.001, Batch : 32
- MSE : 0.46, RMSE : 0.68

Fichier : `pricing-data/results/grid_simple15_lr0.01_m0.9_l20.001_tanh_bs32_learned.json`

## Résultats Principaux

### Impact de la Normalisation

SANS normalisation : MSE ~ 18.5
AVEC normalisation : MSE ~ 0.5
Amélioration : -97%

### Observations du Grid Search

**Architecture** : Plus simple = meilleur
- 7-15-1 : MSE 0.46 (MEILLEUR)
- 7-20-10-1 : MSE 0.49
- 7-30-15-1 : MSE 0.57

**Activation** : Tanh > Relu
- Tanh : MSE moyen 3.27
- Relu : MSE moyen 4.40

**Batch Size** : 32 > 16
- BS 32 : MSE moyen 3.42
- BS 16 : MSE moyen 4.27

**L2 Regularization** : 0.001 > 0.0001
- L2 0.001 : MSE moyen 2.99
- L2 0.0001 : MSE moyen 4.38

## Analyse par Prix

L'erreur est inversement proportionnelle au prix (facteur 54x).

| Range | Échantillons | MAPE |
|-------|--------------|------|
| 0-1 | 285 (19%) | 203% |
| 1-5 | 643 (43%) | 31% |
| 5-10 | 304 (20%) | 12% |
| 10-20 | 213 (14%) | 7% |
| 20-50 | 55 (4%) | 4% |

**Explications** :
1. Sensibilité relative : 0.50 erreur = 100% à 0.50 mais 2.5% à 20
2. Compression numérique : Intervalle 0.1-1 est 10x plus petit que 10-100
3. Normalisation Z-score : Petits prix deviennent négatifs proches de zéro
4. Déséquilibre : 62% des données < 5

**Solutions proposées** :
- Réseaux spécialisés par range
- MAPE loss function
- Transformation logarithmique

## Structure du Projet

```
pricing-data/
├── train.csv (7000)
├── valid.csv (1500)
├── test.csv (1500)
├── experiments/ (configs JSON)
├── results/ (réseaux entraînés)
└── screenshot/ (HiPlot)
```

## Fichiers Importants

**Données** : train.csv, valid.csv, test.csv

**Scripts Python** :
- `generate_grid_search.py` : Génère 56 configs
- `quick_summary.py` : Top 10 réseaux
- `hiplot_analysis.py` : Visualisation interactive
- `analyze_error_by_price.py` : Analyse par range de prix

**Résultats** :
- `all_experiments_summary.csv` : Toutes les expériences
- `error_analysis_*.png` : Graphiques par prix
- `hiplot_visualization.html` : Visualisation interactive
- `*_learned.json` : Réseaux entraînés

**Screenshots** :
- `vue_ensemble.png` : 68 expériences
- `validation_mse_inf_1.png` : Meilleurs réseaux
- `tanh.png` / `relu.png` : Comparaison activations

## Phase 1 : Expériences Baseline (14 expériences)

### Configuration de Référence

**Architecture baseline** :
- Couches : [7 → 20 (Tanh) → 10 (Tanh) → 1 (Identity)]
- Learning rate : 0.01, Momentum : 0.9
- L2 regularization : 1e-4, Batch size : 32

### Résultats Principaux

**Impact de la normalisation : +97.5%**
- Sans normalisation : MSE ≈ 17.30
- Avec normalisation : MSE ≈ 0.43
- **Amélioration moyenne : +48.0%**

### Variantes Testées

| Expérience | MSE (Sans norm) | MSE (Avec norm) | Amélioration |
|------------|----------------|----------------|--------------|
| baseline | 17.30 | 0.43 | +97.5% |
| no_momentum | 17.32 | 0.45 | +97.4% |
| no_reg | 17.27 | 0.46 | +97.3% |
| lr_high (0.05) | 21.52 | 12.00 | +44.2% |
| deep [30,20,10] | 17.43 | 17.26 | +1.0% |
| relu | 17.28 | 17.26 | +0.1% |
| simple [1 layer] | 17.30 | 17.55 | -1.5% |

### Conclusions Phase 1

1. **La normalisation est critique** pour la convergence
2. **Architecture modérée optimale** : 2 couches cachées suffisent
3. **Tanh >> ReLU** pour ce problème de pricing
4. **Learning rate 0.01** est optimal (0.05 dégrade les performances)
5. **Momentum et L2 améliorent légèrement** les résultats

### Transition Phase 1 → Phase 2

Les résultats de la Phase 1 ont orienté le grid search de la Phase 2 :
- **Normalisation systématique** : Toutes les expériences Phase 2 utilisent normalisation
- **Focus sur Tanh** : ReLU et Tanh testés
- **Architectures variées** : De 7-7-1 à 7-50-30-15-1
- **Learning rate fixé** : 0.01 (optimal identifié en Phase 1)
- **Momentum fixé** : 0.9 (meilleur paramètre)
- **Exploration L2** : 0.0001 et 0.001
- **Batch sizes** : 16 et 32

## Pour le Rapport

### Section 4.3 : Analyse par Prix

L'analyse révèle que l'erreur relative est inversement proportionnelle au prix. Pour les prix < 1, le MAPE atteint 203% contre 4% pour les prix > 20.

[Insérer graphique error_analysis_baseline_norm.png]

**Explications** :
- Sensibilité relative
- Compression numérique
- Normalisation Z-score
- Déséquilibre des données

**Pistes d'amélioration** :
- Réseaux spécialisés par range
- MAPE loss function
- Transformation logarithmique

### Graphiques à Inclure

- `error_analysis_baseline_norm.png` (6 subplots)
- `vue_ensemble.png` (HiPlot)
- `validation_mse_inf_1.png` (Top réseaux)
- `tanh.png` vs `relu.png` (Comparaison activations)

### Tableaux

**Tableau 1** : Top 10 réseaux (all_experiments_summary.csv)
**Tableau 2** : Erreur par prix (error_by_price_range.csv)
**Tableau 3** : Stats par architecture/activation/batch/L2

## Fonctionnalités Implémentées

**Hands-on 1** : Forward propagation
**Hands-on 2** : Backpropagation, gradient descent, JSON
**Hands-on 3** : Mini-batch, momentum
**Hands-on 4** : L2 regularization

## Architecture Java

**fr.ensimag.deep.layers** : StandardLayer (fully-connected)
**fr.ensimag.deep.activators** : Tanh, Relu, Sigmoid, Identity
**fr.ensimag.deep.trainer** : NetworkTrainer, QuadraticCostFunction
**fr.ensimag.deep.serialization** : Chargement/sauvegarde JSON

## Commandes Utiles

```bash
# Compiler
mvn clean compile

# Grid search
bash pricing-data/run_grid_search.sh

# Résumé rapide
python3 pricing-data/quick_summary.py

# HiPlot
python3 pricing-data/hiplot_analysis.py

# Analyse par prix
python3 pricing-data/analyze_error_by_price.py
```
