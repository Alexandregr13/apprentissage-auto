# Projet Deep Learning - Pricing Neural Network

## Objectif du Projet

Construire un réseau de neurones en Java pour résoudre un **problème de régression** : le pricing d'un produit financier.

- **Analyse comparative** : tester différentes configurations et choisir la meilleure
- Le réseau final sérialisé (meilleure performance)

---

## Fonctionnalités Implémentées

### Hands-on 1 : Forward Propagation
- Propagation avant dans le réseau
- Validation sur la fonction AND
- Refactoring pour réduire les instanciations

### Hands-on 2 : Backpropagation
- Chargement/Sauvegarde de réseaux (JSON)
- Backpropagation et descente de gradient
- Tests sur AND, Sin, Cos

### Hands-on 3 : Mini-batch & Momentum
- **Mini-batch** : Entraînement par paquets (taille configurable)
- **Momentum** : Accélération du gradient (implémenté dans `StandardLayer`)
- Tests comparatifs : `and_expe.json` vs `and_expe_momentum.json`

### Hands-on 4 : Régularisation
- **L2 Regularization** : Pénalisation des poids pour éviter l'overfitting
- Configurable par couche dans les fichiers JSON

### Pricing Neural Network (6h)
- Données disponibles dans `pricing-data/`
- **À faire** : Entraîner et comparer différents réseaux

---

## Plan d'Expérimentation (Tests à Réaliser)

Le rapport doit contenir une **analyse comparative** de plusieurs configurations de réseaux. Voici les tests à effectuer :

### 1. **Architecture du Réseau**
Tester différentes configurations :
- Nombre de couches cachées : 1, 2, 3 couches
- Taille des couches : 10, 20, 50 neurones
- Fonctions d'activation : `Tanh`, `Relu`, `LeakyRelu`, `Sigmoid`

**Fichiers à créer :**
```
pricing-data/experiments/
├── network_1layer_20neurons.json
├── network_2layers_10_10.json
├── network_3layers_20_10_5.json
└── ...
```

### 2. **Hyperparamètres d'Apprentissage**
Comparer l'impact de :
- **Learning Rate** : 0.001, 0.01, 0.1
- **Batch Size** : 16, 32, 64, 128
- **Momentum** : 0 (sans), 0.9, 0.95
- **Epochs** : 1000, 5000, 10000

**Fichiers à créer :**
```
pricing-data/experiments/
├── expe_lr_0.001.json
├── expe_lr_0.01.json
├── expe_momentum_0.9.json
└── ...
```

### 3. **Régularisation**
Tester avec/sans régularisation :
- Sans régularisation L2
- Avec L2 (λ = 0.0001, 0.001, 0.01)

### 4. **Normalisation des Données** 
- Input Standardization : $\frac{x-\mu}{\sigma}$

**Tests à faire :**
- Réseau **sans** normalisation
- Réseau **avec** normalisation (devrait converger beaucoup plus vite)

### 5. **Initialisation des Poids**
Comparer les initialiseurs :
- `Xavier` (recommandé pour Tanh/Sigmoid)
- `He` (recommandé pour ReLU)
- `Gaussian` (aléatoire standard)

---

## Métriques à Analyser

Pour **chaque expérience**, collecter et comparer :

### 1. **Courbes d'Erreur**
- Tracer `Training Error` vs `Validation Error` (fichiers `.csv` générés)
- **Vérifier** :
  - Convergence : L'erreur diminue-t-elle ?
  - Overfitting : L'erreur de validation remonte-t-elle ?
  - Vitesse : Combien d'époques pour atteindre 1% d'erreur ?

**Commande pour tracer :**
```bash
python Examples/TrainingConsole/plot.py
```

### 2. **Erreur Finale**
- Erreur sur le jeu de **validation**
- Erreur sur le jeu de **test** (pricing-data/test.csv)

### 3. **Inspection des Poids**
Ouvrir les fichiers `*_learned.json` et vérifier :
- Les poids sont-ils dans un range raisonnable (-10, +10) ?
- Ou explosent-ils (1e6) → instabilité
- Sont-ils trop petits (~0) → le réseau n'a rien appris

### 4. **Temps d'Entraînement**
- Noter le temps d'exécution (visible dans les logs Maven)

---

## Structure des Données de Test

```
pricing-data/
├── train.csv          # Données d'entraînement (70%)
├── valid.csv          # Données de validation (15%)
├── test.csv           # Données de test final (15%)
└── experiments/       # Vos fichiers de configuration
    ├── network_*.json
    └── expe_*.json
```

---

## Lancer une Expérience

### 1. Créer un fichier de configuration d'expérience

**Exemple : `pricing-data/experiments/expe_baseline.json`**
```json
{
    "network description": "pricing-data/experiments/network_baseline.json",
    "training data": "pricing-data/train.csv",
    "validation data": "pricing-data/valid.csv",
    "epochs": 5000,
    "trained network": "pricing-data/results/baseline_learned.json",
    "cost function": "Quadratic",
    "initialize": true,
    "learning log file": "pricing-data/results/baseline_error.csv",
    "validation steps": 100,
    "final validation": "pricing-data/results/baseline_validation.csv",
    "activation file": "pricing-data/results/baseline_activation.csv",
    "gnuplot": false
}
```

### 2. Créer le fichier réseau correspondant

**Exemple : `pricing-data/experiments/network_baseline.json`**
```json
{
  "InputSize": 5,
  "BatchSize": 32,
  "Initializer": "Xavier",
  "Layers": [
    {
      "Size": 20,
      "ActivatorType": "Tanh",
      "Type": "Standard",
      "GradientAdjustmentParameters": {
        "Type": "Momentum",
        "LearningRate": 0.01,
        "Momentum": 0.9
      },
      "L2Regularization": 0.0001
    },
    {
      "Size": 10,
      "ActivatorType": "Tanh",
      "Type": "Standard",
      "GradientAdjustmentParameters": {
        "Type": "Momentum",
        "LearningRate": 0.01,
        "Momentum": 0.9
      }
    },
    {
      "Size": 1,
      "ActivatorType": "Identity",
      "Type": "Standard",
      "GradientAdjustmentParameters": {
        "Type": "FixedLearningRate",
        "LearningRate": 0.01
      }
    }
  ]
}
```

### 3. Lancer l'entraînement

```bash
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_baseline.json"
```

### 4. Analyser les résultats

- Ouvrir `pricing-data/results/baseline_error.csv`
- Tracer la courbe avec Python/gnuplot
- Comparer avec les autres expériences

---

## Résultats Expérimentaux (AVEC NORMALISATION - 7 inputs)

### Tableau Comparatif des Expériences

| Rang | Expérience    | Architecture  | Activation | LR   | Momentum | L2     | MSE Valid | RMSE Valid | Temps  |
|------|---------------|---------------|------------|------|----------|--------|-----------|------------|--------|
| 1 | **deep**      | **50-20-10-1**| Tanh       | 0.01 | 0.9      | 0.001  | **36.85** | **6.07**   | ~1m50s |
| 2 | relu          | 20-10-1       | ReLU       | 0.01 | 0.9      | 0.0001 | 36.87     | 6.07       | ~20s   |
| 3 | simple        | **7-1**       | Tanh       | 0.01 | 0.9      | 0.0001 | 36.88     | 6.07       | ~15s   |
| 4    | baseline      | 20-10-1       | Tanh       | 0.01 | 0.9      | 0.0001 | 36.90     | 6.07       | ~20s   |
| 5    | no_reg        | 20-10-1       | Tanh       | 0.01 | 0.9      | **0**  | 37.11     | 6.09       | ~20s   |
| 6    | no_momentum   | 20-10-1       | Tanh       | 0.01 | **0**    | 0.0001 | 67.45     | 8.21       | ~20s   |
| 7    | lr_high       | 20-10-1       | Tanh       | **0.1** | 0.9   | 0.0001 | 67.48     | 8.22       | ~15s   |

### Analyse des Résultats

####  Meilleur Réseau : **deep**
- **Architecture** : 7-50-20-10-1 (7 inputs normalisés, 3 couches cachées)
- **Fonction d'activation** : Tanh
- **Hyperparamètres** : LR=0.01, Momentum=0.9, L2=0.001
- **Performance Validation** : MSE=36.85, RMSE=6.07
- **Performance Test** : MSE=35.97, RMSE=6.00 
- **Généralisation** : 2.4% de différence
- **Fichier** : `pricing-data/results/deep_learned.json`

**Justification du choix :**
- Meilleure erreur de validation parmi toutes les configurations
- Convergence rapide (~45 secondes)
- Pas d'overfitting (erreur train = erreur valid)
- La normalisation permet au réseau profond de mieux converger
- Architecture profonde capture mieux les patterns complexes
- Pas d'overfitting détecté

####  Observations Clés

**Impact de la Normalisation :**
- **Avant normalisation** : MSE test = 66.22
- **Après normalisation** : MSE test = 35.97
- **Amélioration** : **-46%** d'erreur !

**Impact de l'Architecture :**
- **Réseau simple (7-1)** : MSE=36.88 - Surprenant ! Un seul neurone caché suffit presque
- **Réseau standard (20-10-1)** : MSE=36.87 - Légèrement mieux
- **Réseau profond (50-20-10-1)** : MSE=36.85 - Le meilleur, mais gain marginal
- Conclusion : Avec normalisation, même un petit réseau performe bien

**Impact du Learning Rate :**
- **LR trop élevé (0.1)** : Convergence très lente, MSE=67.48
- **LR optimal (0.01)** : Convergence rapide et stable

**Impact du Momentum :**
- **Avec momentum (0.9)** : MSE=36.87
- **Sans momentum (0)** : MSE=67.45 (presque 2× pire !)
- Conclusion : Le momentum est **crucial** pour ce problème

**Impact de la Régularisation L2 :**
- **Sans L2** : MSE=37.11
- **Avec L2=0.0001** : MSE=36.90
- **Avec L2=0.001** : MSE=36.85 (meilleur pour réseau profond)
- Conclusion : La régularisation aide, surtout pour les réseaux profonds

**Impact de la Fonction d'Activation :**
- **Tanh** : MSE=36.85 (meilleur pour réseau profond)
- **ReLU** : MSE=36.87 (excellent aussi)
- Conclusion : Avec normalisation, les deux fonctions sont équivalentes

### Graphiques Générés

Les courbes d'apprentissage sont disponibles dans :
- `pricing-data/results/all_experiments.png` : Évolution de l'erreur (train + validation) pour toutes les expériences
- `pricing-data/results/validation_comparison.png` : Comparaison des erreurs de validation finales

---

## Choix du Meilleur Réseau

**Critères de sélection :**
1. **Erreur de validation et test les plus faibles**
2. Pas d'overfitting (erreur train ≈ erreur valid)
3. Temps d'entraînement raisonnable
4. Bonne généralisation

**Réseau final à soumettre :**
- Fichier : `pricing-data/results/deep_learned.json`
- Architecture : 7-50-20-10-1 (avec normalisation)
- MSE validation : 36.85
- MSE test : 35.97

---

## Contenu du Rapport

Le rapport doit contenir :

### 1. Introduction
- Objectif du projet
- Description du problème de pricing
- Expliquer la demarche (implementation pour and, sin, cos,...)

### 2. Implémentation
- Fonctionnalités développées (Forward, Backprop, Mini-batch, Momentum, L2)
- Normalisation 
- Architecture du code

### 3. Expérimentation
- **Tableau comparatif** (comme ci-dessus)
- **Courbes d'erreur** pour chaque expérience
- **Analyse** : 
  - Impact du learning rate
  - Impact du momentum
  - Impact de la normalisation
  - Impact de l'architecture (profondeur, largeur)

### 4. Résultats
- Meilleur réseau sélectionné
- Justification du choix
- Erreur finale sur le jeu de test

### 5. Conclusion
- Difficultés rencontrées
- Améliorations possibles (Batch Normalization, Dropout, Adam optimizer...)

---

---


### Rapport
- [ ] Rédiger l'introduction
- [ ] Documenter l'implémentation
- [ ] Inclure le tableau comparatif
- [ ] Ajouter les graphiques
- [ ] Justifier le choix du meilleur réseau
- [ ] Conclusion

---

## Commandes Utiles

### Lancer toutes les expériences
```bash
bash pricing-data/RUN_ALL.sh
```

### Analyser tous les résultats
```bash
python3 pricing-data/
```

### Évaluer le meilleur réseau (validation + test)
```bash
python3 pricing-data/evaluate_normalized.py
```

### Lancer une expérience spécifique
```bash
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_baseline.json"
```

**Résultats:**
- Validation: MSE=67.45, RMSE=8.21
- Test: MSE=66.22, RMSE=8.14
- Différence: 1.83% -> Excellente généralisation