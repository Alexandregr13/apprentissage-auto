# Guide d'Installation et de Test

Ce guide permet de vérifier que le projet fonctionne correctement depuis zéro.


## Étape 1 : Compilation

```bash
mvn clean compile
```

## Étape 2 : Test des Exemples de Base

### Test AND (hardcodé)
```bash
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.examples.And" -q
```

### Test AND depuis fichier JSON
```bash
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.examples.AndFromFile" -q
```

## Étape 3 : Vérification des Données Pricing

```bash
ls -lh pricing-data/*.csv
```

**Fichiers attendus :**
- `pricing-data-inputs.csv` (1.2M)
- `pricing-data-outputs.csv` (177K)
- `train.csv` (956K)
- `valid.csv` (205K)
- `test.csv` (205K)

Si les fichiers train/valid/test sont absents :
```bash
python3 pricing-data/prepare_data.py
```

## Étape 4 : Test d'une Expérience Simple

```bash
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_baseline.json" -q
```

**Durée :** ~1-2 minutes  
**Fichiers générés dans `pricing-data/results/` :**
- `baseline_error.csv` : Évolution de l'erreur
- `baseline_learned.json` : Réseau entraîné
- `baseline_validation.csv` : Prédictions sur les données de validation

## Étape 5 : Lancement Complet de Toutes les Expériences

```bash
bash pricing-data/RUN_ALL.sh
```
(ou depuis pricing-data/ : `bash RUN_ALL.sh`)

**Durée totale :** ~20-30 minutes  
**Contenu :**
1. 7 expériences sans normalisation
2. 7 expériences avec normalisation
3. Génération automatique de l'analyse comparative

**Fichiers générés :**
- `results/normalization_comparison.csv` : Comparaison des résultats
- `results/normalization_impact.png` : Graphiques comparatifs
- `results/*_error.csv` : Courbes d'erreur pour chaque expérience
- `results/*_learned.json` : Réseaux entraînés

## Étape 6 : Analyse des Résultats

### Comparaison normalisation
```bash
python3 pricing-data/compare_normalization.py
```

### Courbes de convergence
```bash
python3 pricing-data/plot_convergence.py
```
Génère 14 graphiques `results/*_convergence.png`

### Inspection des poids
```bash
python3 pricing-data/inspect_weights.py
```

### Analyse complète
```bash
python3 pricing-data/complete_analysis.py
```
Génère `results/complete_analysis.csv` avec toutes les métriques.


## Documentation

- `README.md` : Description générale du projet
- `pricing-data/GUIDE.md` : Analyse détaillée des expériences et résultats
- `INSTALLATION.md` : Ce guide

## Résultats Attendus

**Meilleur réseau : no_reg_norm**
- MSE : 0.41
- RMSE : 0.49
- Amélioration vs sans normalisation : +97.6%

Pour plus de détails, consulter `pricing-data/GUIDE.md`.
