# Projet Deep Learning - Pricing Neural Network

## 📋 Objectif du Projet

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

### 4. **Normalisation des Données** ⚠️ À IMPLÉMENTER
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

## 📈 Tableau Comparatif (À Remplir)

| Expérience | Architecture | LR | Momentum | L2 | Normalisation | Erreur Train | Erreur Valid | Temps |
|------------|--------------|----|-----------|----|---------------|--------------|--------------|-------|
| baseline   | 20-10-1 Tanh | 0.01 | 0.9 | 0.0001 | Non | ? | ? | ? |
| lr_high    | 20-10-1 Tanh | 0.1 | 0.9 | 0.0001 | Non | ? | ? | ? |
| no_momentum| 20-10-1 Tanh | 0.01 | 0 | 0.0001 | Non | ? | ? | ? |
| normalized | 20-10-1 Tanh | 0.01 | 0.9 | 0.0001 | **Oui** | ? | ? | ? |
| deep_net   | 50-20-10-1 Tanh | 0.01 | 0.9 | 0.001 | Oui | ? | ? | ? |

---

## 🎯 Choix du Meilleur Réseau

**Critères de sélection :**
1. **Erreur de validation la plus faible**
2. Pas d'overfitting (erreur train environ egale à l'erreur valid)
3. Temps d'entraînement raisonnable
4. Stabilité (poids cohérents)

**Réseau final à soumettre :**
- Le fichier `*_learned.json` avec les meilleures performances
- Accompagné d'une justification dans le rapport

---

## Contenu du Rapport

Le rapport doit contenir :

### 1. Introduction
- Objectif du projet
- Description du problème de pricing

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

## 🔧 TODO List

### Implémentation
- [ ] **Input Standardization** 
  - Calculer μ et σ sur les données d'entraînement
  - Normaliser train, valid et test avec ces valeurs
  
### Expérimentation
- [ ] Créer le dossier `pricing-data/experiments/`
- [ ] Créer le dossier `pricing-data/results/`
- [ ] Générer les fichiers de configuration pour chaque test
- [ ] Lancer toutes les expériences
- [ ] Collecter les résultats dans un tableau Excel/CSV

### Analyse
- [ ] Tracer toutes les courbes d'erreur
- [ ] Comparer les performances
- [ ] Inspecter les poids du meilleur réseau
- [ ] Tester le réseau final sur `test.csv`

### Rapport
- [ ] Rédiger l'introduction
- [ ] Documenter l'implémentation
- [ ] Inclure le tableau comparatif
- [ ] Ajouter les graphiques
- [ ] Justifier le choix du meilleur réseau
- [ ] Conclusion

