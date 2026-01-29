#!/bin/bash

# Script pour lancer toutes les expériences (avec et sans normalisation)

set -e  

# Déterminer le répertoire du projet (racine avec pom.xml)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Se positionner à la racine du projet
cd "$PROJECT_ROOT"

EXPERIMENTS=("baseline" "lr_high" "no_momentum" "deep" "relu" "no_regularization" "simple")
TOTAL=${#EXPERIMENTS[@]}

echo "  PARTIE 1/3 : EXPÉRIENCES SANS NORMALISATION"
echo ""

for i in "${!EXPERIMENTS[@]}"; do
  exp="${EXPERIMENTS[$i]}"
  current=$((i + 1))
  
  echo "[$current/$TOTAL] Lancement: $exp (SANS normalisation)"
  mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
    -Dexec.args="-x pricing-data/experiments/expe_${exp}.json" -q
  echo "  $exp terminé"
  echo ""
done

echo " Partie 1 terminée : Toutes les expériences sans normalisation"
echo ""


echo "  PARTIE 2/3 : EXPÉRIENCES AVEC NORMALISATION"
echo ""

for i in "${!EXPERIMENTS[@]}"; do
  exp="${EXPERIMENTS[$i]}"
  current=$((i + 1))
  
  echo "[$current/$TOTAL]  Lancement: $exp (AVEC normalisation)"
  mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
    -Dexec.args="-x pricing-data/experiments/expe_${exp}_norm.json" -q
  echo "   $exp (norm) terminé"
  echo ""
done

echo " Partie 2 terminée : Toutes les expériences avec normalisation"
echo ""

echo "  PARTIE 3/3 : ANALYSE ET COMPARAISON"
echo ""

echo " Génération du rapport de comparaison..."
python3 pricing-data/compare_normalization.py

echo ""

echo ""
echo " Résultats disponibles :"
echo "  - pricing-data/results/*_error.csv (erreurs d'entraînement)"
echo "  - pricing-data/results/*_learned.json (réseaux entraînés)"
echo "  - pricing-data/results/normalization_comparison.csv (comparaison)"
echo "  - pricing-data/results/normalization_impact.png (graphiques)"
echo "  - pricing-data/GUIDE.md (guide complet)"
echo ""
echo " Prochaines étapes :"
echo "  - Consulter GUIDE.md pour l'analyse détaillée"
echo "  - Visualiser normalization_impact.png"
echo "  - Choisir le meilleur réseau pour le déploiement"
echo ""
