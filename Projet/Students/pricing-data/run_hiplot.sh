#!/bin/bash
# Lancement HiPlot

if ! python3 -c "import hiplot" 2>/dev/null; then
    echo "HiPlot non installé"
    read -p "Installer maintenant? (o/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Oo]$ ]]; then
        pip install hiplot || exit 1
    else
        exit 1
    fi
fi

python3 pricing-data/hiplot_analysis.py
