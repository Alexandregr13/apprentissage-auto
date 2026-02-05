# Analyse de l'Erreur par Prix - Guide Rapide

## Résultat Principal

**L'erreur est inversement proportionnelle au prix** :
- Prix < 1€ : MAPE = 203%
- Prix > 20€ : MAPE = 4%
- **Facteur × 54**

## Fichiers Importants

**Documentation** : [ANALYSE_ERREUR_PAR_PRIX.md](ANALYSE_ERREUR_PAR_PRIX.md) - Tout est dedans

**Scripts** :
- `analyze_error_by_price.py` - Analyse l'erreur par range
- `split_by_price_range.py` - Crée les réseaux spécialisés

**Résultats** :
- `results/error_analysis_baseline_norm.png` - Graphique principal
- `results/error_by_price_range.csv` - Données pour le rapport

## Utilisation

### 1. Analyser l'erreur par prix
```bash
python3 pricing-data/analyze_error_by_price.py
```

### 2. Créer les réseaux spécialisés (optionnel)
```bash
python3 pricing-data/split_by_price_range.py
```

Puis lancer les entraînements affichés par le script.

## Pour le Rapport

Voir [ANALYSE_ERREUR_PAR_PRIX.md](ANALYSE_ERREUR_PAR_PRIX.md), section "Pour le Rapport"

1. Copier la section 4.3
2. Inclure le graphique error_analysis_baseline_norm.png
3. Ajouter le tableau des résultats
4. Mentionner l'observation du prof

C'est tout ! ✅
