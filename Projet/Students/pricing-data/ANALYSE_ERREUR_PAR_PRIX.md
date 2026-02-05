# Analyse de l'Erreur en Fonction du Prix

## Confirmation du Feedback du Prof

**Observation** : "Quand le prix est bas, c'est plus compliqué à estimer"

**Résultat** : CONFIRMÉ

### Résultats par Range de Prix

| Range Prix | Échantillons | Prix Moyen | MAPE (%) | RMSE | MAE |
|------------|--------------|------------|----------|------|-----|
| 0-1€       | 285 (19%)    | 0.53€      | 203.74   | 0.77 | 0.62 |
| 1-5€       | 643 (43%)    | 2.60€      | 30.96    | 0.96 | 0.76 |
| 5-10€      | 304 (20%)    | 7.21€      | 12.11    | 1.10 | 0.86 |
| 10-20€     | 213 (14%)    | 13.54€     | 6.89     | 1.12 | 0.88 |
| 20-50€     | 55 (4%)      | 26.18€     | 3.77     | 1.24 | 1.00 |

**L'erreur relative est 54× plus élevée pour les prix < 1€ que pour les prix > 20€**

## Explications

### 1. Sensibilité Relative
```
Prix = 0.50€, Prédiction = 1.00€ → Erreur = 100%
Prix = 20.00€, Prédiction = 21.00€ → Erreur = 5%
```

### 2. Compression Numérique
L'intervalle 0.1-1€ est 10× plus petit que 10-100€, rendant les distinctions plus difficiles.

### 3. Normalisation Z-Score
```python
prix_norm = (prix - mean) / std  # mean=5.4, std=5.9
Prix 0.5€ → normalisé = -0.83
Prix 25€ → normalisé = +3.32
```
Les petits prix deviennent des valeurs négatives proches de zéro.

### 4. Déséquilibre des Données
62% des données ont un prix < 5€, donc le réseau est biaisé vers les petits prix.

## Solutions Proposées

### Solution 1 : Réseaux Spécialisés par Range

Entraîner 3 réseaux experts au lieu d'un généraliste :

**Réseau Petits Prix (0-5€)**
- Architecture : 7-50-30-15-1 (plus profonde)
- Données : 4422 échantillons (train_small.csv)
- Objectif : MAPE < 20% (vs 31% actuel)

**Réseau Prix Moyens (5-15€)**
- Architecture : 7-20-10-1 (standard)
- Données : 2031 échantillons (train_medium.csv)

**Réseau Grands Prix (15-50€)**
- Architecture : 7-10-1 (simple, éviter overfitting)
- Données : 547 échantillons (train_large.csv)

**Commandes pour entraîner** :
```bash
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_small_prices.json"

mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_medium_prices.json"

mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_large_prices.json"
```

### Solution 2 : MAPE Loss Function

Remplacer MSE par MAPE pour pénaliser directement l'erreur relative :
```java
public class MAPECostFunction implements ICostFunction {
    @Override
    public double cost(DataMatrix prediction, DataMatrix target) {
        double sum = 0.0;
        for (int i = 0; i < prediction.getRows(); i++) {
            double pred = prediction.get(i, 0);
            double true_val = target.get(i, 0);
            sum += Math.abs(pred - true_val) / Math.abs(true_val);
        }
        return sum / prediction.getRows() * 100;
    }
}
```

### Solution 3 : Transformation Logarithmique

Travailler dans l'espace log pour équilibrer les échelles :
```python
prix_transformé = log(prix + 1)
# Après prédiction
prix_final = exp(prix_prédit) - 1
```

## Pour le Rapport

### Section à Ajouter (4.3)

**4.3 Analyse de l'Erreur en Fonction du Prix**

Comme suggéré par l'enseignant, nous avons analysé l'évolution de l'erreur en fonction du prix. Les résultats révèlent une corrélation inversée entre le prix et l'erreur relative.

[Insérer graphique : error_analysis_baseline_norm.png]

[Insérer tableau ci-dessus]

**Observations** :
- L'erreur relative (MAPE) diminue de 203% à 4% quand le prix augmente de 0.5€ à 25€
- L'erreur absolue augmente légèrement (0.62€ à 1.00€) mais l'erreur relative diminue drastiquement

**Explications** :
1. Sensibilité relative : Une erreur de 0.50€ est catastrophique à 0.50€ mais négligeable à 20€
2. Compression numérique : Les petits prix sont dans un intervalle 10× plus petit
3. Normalisation : Les petits prix deviennent négatifs et proches de zéro après normalisation
4. Déséquilibre : 62% des données ont un prix < 5€

**Pistes d'amélioration** :
Pour améliorer la performance sur les petits prix, nous proposons :
- Réseaux spécialisés par range de prix
- MAPE loss function pour pénaliser l'erreur relative
- Transformation logarithmique des prix

Ces approches constituent des perspectives pour des travaux futurs.

## Fichiers Générés

**Scripts** :
- `analyze_error_by_price.py` - Analyse l'erreur par range
- `split_by_price_range.py` - Crée les données et configs pour réseaux spécialisés

**Résultats** :
- `results/error_analysis_baseline_norm.png` - Graphique principal (6 subplots)
- `results/error_analysis_baseline_norm_binned.png` - Évolution continue
- `results/error_by_price_range.csv` - Données brutes

**Données divisées** :
- `train_small.csv` / `valid_small.csv` (4422 / 928 échantillons)
- `train_medium.csv` / `valid_medium.csv` (2031 / 456 échantillons)
- `train_large.csv` / `valid_large.csv` (547 / 116 échantillons)

## Screenshots HiPlot

**vue_ensemble.png** : Vue des 14 expériences, montre l'impact de la normalisation

**normalized_yes.png** vs **normalized_no.png** :
- Avec normalisation : MSE ~ 0.5
- Sans normalisation : MSE ~ 18.5
- Amélioration : -97%

**validation_inf_1.png** : Filtre MSE < 1, montre que seules les expériences normalisées réussissent

**learning_rate_0.1.png** : LR trop élevé cause divergence (MSE = 5.42 vs 0.49)

## Commandes Utiles

```bash
# Analyser l'erreur par prix
python3 pricing-data/analyze_error_by_price.py

# Créer les réseaux spécialisés (données + configs)
python3 pricing-data/split_by_price_range.py

# Entraîner les réseaux spécialisés (optionnel)
# Voir commandes dans "Solution 1" ci-dessus
```

## Checklist Rapport

- [ ] Copier la section 4.3 dans le rapport
- [ ] Inclure le graphique error_analysis_baseline_norm.png
- [ ] Créer le tableau à partir des résultats ci-dessus
- [ ] Mentionner l'observation du prof
- [ ] (Optionnel) Entraîner les réseaux spécialisés et ajouter section 4.4
