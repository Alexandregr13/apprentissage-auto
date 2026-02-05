# Projet Deep Learning - Pricing Neural Network

## Installation et Compilation

```bash
mvn clean compile
```

## Structure du Projet

### Données
```
pricing-data/
├── train.csv (7000 échantillons)
├── valid.csv (1500 échantillons)
├── test.csv (1500 échantillons)
├── experiments/ (configurations JSON)
└── results/ (réseaux entraînés, courbes, métriques)
```

### Tests de Base

**Test AND hardcodé:**
```bash
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.examples.And" -q
```

**Test AND depuis JSON:**
```bash
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.examples.AndFromFile" -q
```

## Expérimentations

### Lancer une Expérience Simple

```bash
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_baseline_norm.json"
```

Fichiers générés:
- `*_error.csv` : Evolution de l'erreur
- `*_learned.json` : Réseau entraîné avec poids
- `*_validation.csv` : Prédictions sur validation
- `*_normalizer.json` : Paramètres de normalisation

### Expériences Complètes (14 configurations)

```bash
bash pricing-data/RUN_ALL.sh
```

Durée: 20-30 minutes

Tests 7 configurations x 2 (avec/sans normalisation):
- baseline (Tanh, momentum 0.9, L2 0.0001)
- lr_high (LR 0.1)
- no_momentum (momentum 0)
- deep (3 couches cachées)
- relu (activation ReLU)
- no_regularization (L2 = 0)
- simple (1 couche cachée)

### Grid Search (56 configurations)

Teste systématiquement architectures et hyperparamètres:

```bash
# Générer les configurations
python3 pricing-data/generate_grid_search.py

# Lancer les entraînements (23 min)
bash pricing-data/run_grid_search.sh
```

Architectures testées:
- tiny (7)
- simple (10)
- simple15 (15)
- standard (20-10)
- medium (30-15)
- deep (30-20-10)
- very_deep (50-30-15)

Paramètres:
- Activations: Tanh, Relu
- L2 regularization: 0.0001, 0.001
- Batch sizes: 16, 32

## Analyse des Résultats

### Résumé Rapide

```bash
python3 pricing-data/quick_summary.py
```

Affiche:
- Top 10 meilleurs réseaux
- Stats par architecture
- Stats par activation (Tanh vs Relu)
- Stats par batch size et L2
- Meilleur réseau global

### Visualisation HiPlot (interactive)

```bash
pip install hiplot
python3 pricing-data/hiplot_analysis.py
```

Ouvre `pricing-data/results/hiplot_visualization.html`

Permet de:
- Explorer les 70 expériences (14 + 56 grid)
- Filtrer par validation_mse < 1
- Identifier patterns entre hyperparamètres
- Comparer avec/sans normalisation

### Analyse de l'Erreur par Prix

```bash
python3 pricing-data/analyze_error_by_price.py
```

Génère:
- `error_analysis_*.png` : 6 graphiques d'analyse
- `error_by_price_range.csv` : Données par range de prix

**Découverte clé**: L'erreur relative est inversement proportionnelle au prix

| Range Prix | Échantillons | MAPE |
|------------|--------------|------|
| 0-1 | 285 (19%) | 203% |
| 1-5 | 643 (43%) | 31% |
| 5-10 | 304 (20%) | 12% |
| 10-20 | 213 (14%) | 7% |
| 20-50 | 55 (4%) | 4% |

L'erreur est 54x plus élevée pour les prix bas.

**Explications**:
1. Sensibilité relative: 0.50 erreur = 100% à 0.50 prix, mais 2.5% à 20 prix
2. Compression numérique: Intervalle 0.1-1 est 10x plus petit que 10-100
3. Normalisation Z-score: Petits prix deviennent négatifs proches de zéro
4. Déséquilibre données: 62% des données ont prix < 5

**Solutions proposées**:
- Réseaux spécialisés par range de prix
- MAPE loss function (pénalise erreur relative)
- Transformation logarithmique des prix

### Réseaux Spécialisés par Range de Prix

Divise les données et crée 3 réseaux experts:

```bash
python3 pricing-data/split_by_price_range.py
```

Crée:
- `train_small.csv` (4422 échantillons, prix 0-5)
- `train_medium.csv` (2031 échantillons, prix 5-15)
- `train_large.csv` (547 échantillons, prix 15-50)

Entraîner les réseaux spécialisés:
```bash
mvn exec:java -Dexec.args="-x pricing-data/experiments/expe_small_prices.json"
mvn exec:java -Dexec.args="-x pricing-data/experiments/expe_medium_prices.json"
mvn exec:java -Dexec.args="-x pricing-data/experiments/expe_large_prices.json"
```

## Résultats Principaux

### Impact de la Normalisation

SANS normalisation:
- MSE moyen: 18.5
- Tous les réseaux échouent (MSE 18-19)

AVEC normalisation:
- MSE moyen: 0.6
- Amélioration: -97%
- Meilleur réseau: MSE 0.49

### Meilleurs Réseaux (avec normalisation)

1. baseline_norm (MSE 0.49, RMSE 0.70)
   - Architecture: 7-20-10-1
   - Activation: Tanh
   - LR 0.01, Momentum 0.9, L2 0.0001

2. no_regularization_norm (MSE 0.49)
   - Même architecture, L2 = 0
   - Plus simple, performance identique

3. relu_norm (MSE 0.51)
   - Architecture: 7-20-10-1
   - Activation: Relu

### Observations

**Normalisation**: LE facteur décisif (97% amélioration)

**Architecture**: Impact marginal avec normalisation
- Simple (7-7-1): MSE 0.69
- Standard (7-20-10-1): MSE 0.49
- Deep (7-50-20-10-1): MSE 0.68
Conclusion: Pas besoin de complexité excessive

**Momentum**: Devient optionnel avec normalisation
- Avec momentum 0.9: MSE 0.49
- Sans momentum: MSE 0.51 (+4% seulement)

**Régularisation L2**: Inutile voire contre-productive
- L2 = 0: MSE 0.49
- L2 = 0.0001: MSE 0.49
- L2 = 0.001: MSE 0.68

**Learning Rate**: Toujours critique
- LR 0.01: MSE 0.49
- LR 0.1: MSE 5.42 (10x pire)

**Activation**: Tanh légèrement meilleur que Relu
- Tanh: MSE 0.49
- Relu: MSE 0.51

## Pour le Rapport

### Fichiers à Inclure

**Graphiques**:
- `pricing-data/results/error_analysis_baseline_norm.png` (6 subplots)
- `pricing-data/results/normalization_impact.png`
- Screenshots HiPlot (vue_ensemble.png, normalized_yes/no.png)

**Tableaux**:
- Tableau comparatif des 14 expériences (GUIDE.md)
- Tableau erreur par range de prix (error_by_price_range.csv)

### Section Rapport: Analyse par Prix

**4.3.1 Observation**

L'analyse révèle que l'erreur relative est inversement proportionnelle au prix. Pour les prix < 1, le MAPE atteint 203% contre seulement 4% pour les prix > 20, soit un facteur 54x.

[Insérer graphique error_analysis_baseline_norm.png]
[Insérer tableau des résultats par range]

**4.3.2 Explications**

1. Sensibilité relative: Une erreur absolue de 0.50 représente 100% du prix à 0.50 mais seulement 2.5% à 20
2. Compression numérique: L'intervalle 0.1-1 est 10x plus petit que 10-100
3. Normalisation Z-score: Les petits prix deviennent des valeurs négatives proches de zéro après normalisation
4. Déséquilibre des données: 62% des échantillons ont un prix < 5

**4.3.3 Pistes d'Amélioration**

- Réseaux spécialisés par range de prix
- MAPE loss function pour pénaliser directement l'erreur relative
- Transformation logarithmique pour équilibrer les échelles

Ces approches constituent des perspectives pour travaux futurs.

## Architecture du Code Java

### Packages Principaux

**fr.ensimag.deep.layers**: Couches du réseau
- AbstractLayer: Classe de base
- StandardLayer: Couche fully-connected avec momentum et L2

**fr.ensimag.deep.activators**: Fonctions d'activation
- Identity, Tanh, Sigmoid, Relu, LeakyRelu

**fr.ensimag.deep.trainer**: Entraînement
- NetworkTrainer: Gère le training loop, mini-batches
- costFunction: QuadraticCostFunction

**fr.ensimag.deep.serialization**: Chargement/sauvegarde JSON
- NetworkDescription: Représentation JSON du réseau
- LayerFactory, ActivatorFactory: Création depuis JSON

### Fonctionnalités Implémentées

**Hands-on 1**: Forward propagation
**Hands-on 2**: Backpropagation, gradient descent, serialization JSON
**Hands-on 3**: Mini-batch, momentum
**Hands-on 4**: L2 regularization

## Commandes Utiles

```bash
# Compiler
mvn clean compile

# Lancer expérience
mvn exec:java -Dexec.args="-x pricing-data/experiments/expe_NAME.json"

# Grid search complet
bash pricing-data/run_grid_search.sh

# Analyse rapide
python3 pricing-data/quick_summary.py

# HiPlot
python3 pricing-data/hiplot_analysis.py

# Analyse par prix
python3 pricing-data/analyze_error_by_price.py

# Réseaux spécialisés
python3 pricing-data/split_by_price_range.py
```

## Documentation Détaillée

Voir pricing-data/ANALYSE_ERREUR_PAR_PRIX.md pour:
- Analyse complète de l'erreur par prix
- Solutions détaillées (réseaux spécialisés, MAPE loss, log transform)
- Code Java pour MAPE cost function
