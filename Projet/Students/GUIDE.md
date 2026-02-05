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

## Résultats Expérimentaux (14 expériences - 7 inputs)

### Tableau Comparatif des Expériences (AVEC normalisation)

| Rang | Expérience | Architecture | Activation | LR | Momentum | L2 | **MSE Valid** | RMSE | Temps |
|------|------------|--------------|------------|----|----------|----|---------------|------|-------|
| 1 | **baseline_norm** | 7-20-10-1 | Tanh | 0.01 | 0.9 | 0.0001 | **0.49** | 0.70 | ~20s |
| 2 | **no_regularization_norm** | 7-20-10-1 | Tanh | 0.01 | 0.9 | **0** | **0.49** | 0.70 | ~20s |
| 3 | relu_norm | 7-20-10-1 | ReLU | 0.01 | 0.9 | 0.0001 | **0.51** | 0.72 | ~29s |
| 4 | no_momentum_norm | 7-20-10-1 | Tanh | 0.01 | **0** | 0.0001 | **0.51** | 0.72 | ~20s |
| 5 | deep_norm | 7-50-20-10-1 | Tanh | 0.01 | 0.9 | 0.001 | **0.68** | 0.83 | ~1m48s |
| 6 | simple_norm | 7-7-1 | Tanh | 0.01 | 0.9 | 0.0001 | **0.69** | 0.83 | ~10s |
| 7 | lr_high_norm | 7-20-10-1 | Tanh | **0.1** | 0.9 | 0.0001 | 5.42 | 2.33 | ~20s |

### Tableau Comparatif (SANS normalisation)

| Rang | Expérience | MSE Valid | RMSE |
|------|------------|-----------|------|
| 1 | relu | 18.43 | 4.29 |
| 2 | deep | 18.43 | 4.29 |
| 3 | no_momentum | 18.44 | 4.29 |
| 4 | baseline | 18.46 | 4.30 |
| 5 | no_regularization | 18.49 | 4.30 |
| 6 | lr_high | 18.73 | 4.33 |
| 7 | simple | 19.32 | 4.40 |

**Toutes les expériences sans normalisation ont des MSE entre 18-19 (très cohérent).**

### Analyse des Résultats

#### 🏆 Meilleur Réseau : **baseline_norm** ou **no_regularization_norm**

Deux réseaux ex-aequo avec MSE = 0.49 :

**Configuration recommandée : no_regularization_norm**
- **Architecture** : 7-20-10-1 (2 couches cachées)
- **Activation** : Tanh
- **Hyperparamètres** : LR=0.01, Momentum=0.9, L2=0 (pas de régularisation)
- **Performance Validation** : MSE=0.49, RMSE=0.70
- **Temps d'entraînement** : ~20 secondes
- **Fichier** : `pricing-data/results/no_regularization_norm_learned.json`

**Justification :**
- Performance identique à baseline_norm
- Plus simple (pas de régularisation L2)
- Principe du rasoir d'Ockham : préférer la solution la plus simple

#### 📊 Observations Clés (Découvertes via HiPlot)

**1. Impact MASSIF de la Normalisation ⭐⭐⭐**
- **Sans normalisation** : MSE moyen = 18.5
- **Avec normalisation** : MSE moyen = 0.6
- **Amélioration** : **-97%** d'erreur !
- **Conclusion** : La normalisation est **LE facteur décisif**. Sans elle, impossible d'obtenir de bons résultats.

**2. Architecture : Impact Marginal (avec normalisation)**
- **Simple (7-7-1)** : MSE = 0.69
- **Standard (7-20-10-1)** : MSE = 0.49
- **Deep (7-50-20-10-1)** : MSE = 0.68
- **Conclusion** : Avec normalisation, même un réseau minimal (1 couche cachée) performe très bien. Pas besoin de complexité excessive.

**3. Momentum : Devient Optionnel (avec normalisation)**
- **Avec momentum (0.9)** : MSE = 0.49
- **Sans momentum (0)** : MSE = 0.51 (seulement +4%)
- **Conclusion** : Contrairement aux attentes, le momentum n'est plus critique avec normalisation. La normalisation stabilise l'optimisation.

**4. Régularisation L2 : Inutile (avec normalisation)**
- **Sans L2 (0)** : MSE = 0.49
- **Avec L2 (0.0001)** : MSE = 0.49
- **Avec L2 (0.001)** : MSE = 0.68
- **Conclusion** : La régularisation n'apporte rien, voire dégrade légèrement. La normalisation prévient déjà l'overfitting.

**5. Activation : Tanh ≈ ReLU (avec normalisation)**
- **Tanh** : MSE = 0.49
- **ReLU** : MSE = 0.51
- **Conclusion** : Les deux fonctions sont équivalentes avec normalisation.

**6. Learning Rate : Toujours Critique**
- **LR = 0.01** : MSE = 0.49-0.69 ✅
- **LR = 0.1** : MSE = 5.42 ❌ (10× pire)
- **Conclusion** : Même avec normalisation, un LR trop élevé cause divergence.

### Graphiques et Visualisations

**HiPlot (interactif)** :
- `pricing-data/results/hiplot_visualization.html` : Exploration interactive de tous les hyperparamètres
- Permet de filtrer, comparer et identifier visuellement les patterns

**Courbes d'apprentissage** :
- `pricing-data/results/*_convergence.png` : Évolution de l'erreur pour chaque expérience
- `pricing-data/results/normalization_impact.png` : Comparaison avec/sans normalisation

---

## Visualisation Interactive avec HiPlot

**HiPlot** (Facebook Research) permet d'explorer visuellement les 14 expériences avec de multiples hyperparamètres.

### Utilisation

```bash
pip install hiplot
bash pricing-data/run_hiplot.sh
```

Ouvre `pricing-data/results/hiplot_visualization.html` dans le navigateur.

### Interface

**Parallel Coordinates Plot:**
- Chaque ligne = une expérience
- Chaque axe = un hyperparamètre ou métrique
- Cliquer-glisser sur un axe pour filtrer

**Axes importants:** `validation_mse`, `normalized`, `learning_rate`, `momentum`, `architecture`

**Découvertes clés via HiPlot:**
- Impact massif de la normalisation (-97% d'erreur)
- Momentum optionnel avec normalisation
- Régularisation L2 inutile
- Architecture simple suffit

Voir `HIPLOT_README.md` pour plus de détails.

---

## Choix du Meilleur Réseau

**Critères de sélection :**
1. **Erreur de validation la plus faible**
2. Simplicité (rasoir d'Ockham)
3. Temps d'entraînement raisonnable
4. Pas d'overfitting

**Réseau final recommandé :**
- **Fichier** : `pricing-data/results/no_regularization_norm_learned.json`
- **Architecture** : 7-20-10-1 (avec normalisation)
- **Hyperparamètres** : LR=0.01, Momentum=0.9, L2=0
- **MSE validation** : 0.49
- **RMSE** : 0.70
- **Temps** : 20 secondes

**Alternative (identique) :**
- `pricing-data/results/baseline_norm_learned.json` (MSE=0.49)

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
# Comparaison avec/sans normalisation
python3 pricing-data/compare_normalization.py

# Analyse complète
python3 pricing-data/complete_analysis.py

# Tracer les courbes de convergence
python3 pricing-data/plot_convergence.py

# Inspecter les poids (détecter explosion/neurones morts)
python3 pricing-data/inspect_weights.py
```

### Visualisation interactive HiPlot
```bash
# Lancer HiPlot (installe automatiquement si nécessaire)
bash pricing-data/run_hiplot.sh

# Ou directement
python3 pricing-data/hiplot_analysis.py
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