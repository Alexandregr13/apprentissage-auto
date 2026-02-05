#!/usr/bin/env python3
"""Analyse de l'erreur en fonction du prix"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_predictions(validation_file, true_values_file):
    """Analyse les prédictions et calcule l'erreur par range de prix"""

    # Charger données
    true_data = pd.read_csv(true_values_file, header=None)
    pred_data = pd.read_csv(validation_file, header=None)

    true_prices = true_data.iloc[:, -1].values
    predicted_prices = pred_data.iloc[:, -1].values

    # Créer DataFrame avec vraies valeurs et prédictions
    df = pd.DataFrame({
        'true_price': true_prices,
        'predicted_price': predicted_prices,
        'absolute_error': np.abs(true_prices - predicted_prices),
        'relative_error': np.abs(true_prices - predicted_prices) / true_prices * 100,
        'squared_error': (true_prices - predicted_prices) ** 2
    })

    # Définir les ranges de prix
    price_ranges = [
        (0, 1, "Très bas (0-1)"),
        (1, 5, "Bas (1-5)"),
        (5, 10, "Moyen (5-10)"),
        (10, 20, "Élevé (10-20)"),
        (20, 50, "Très élevé (20-50)")
    ]

    print("\n" + "=" * 80)
    print("ANALYSE DE L'ERREUR PAR RANGE DE PRIX")
    print("=" * 80)

    results = []

    for min_price, max_price, label in price_ranges:
        mask = (df['true_price'] >= min_price) & (df['true_price'] < max_price)
        subset = df[mask]

        if len(subset) == 0:
            continue

        count = len(subset)
        mean_true_price = subset['true_price'].mean()
        mse = subset['squared_error'].mean()
        rmse = np.sqrt(mse)
        mae = subset['absolute_error'].mean()
        mape = subset['relative_error'].mean()
        median_abs_error = subset['absolute_error'].median()

        results.append({
            'range': label,
            'min_price': min_price,
            'max_price': max_price,
            'count': count,
            'mean_price': mean_true_price,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE_%': mape,
            'median_abs_error': median_abs_error
        })

        print(f"\n{label}")
        print(f"  Nombre d'échantillons: {count}")
        print(f"  Prix moyen: {mean_true_price:.2f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE (Mean Absolute Error): {mae:.4f}")
        print(f"  MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
        print(f"  Erreur absolue médiane: {median_abs_error:.4f}")

    # Statistiques globales
    print("\n" + "=" * 80)
    print("STATISTIQUES GLOBALES")
    print("=" * 80)
    global_mse = df['squared_error'].mean()
    global_rmse = np.sqrt(global_mse)
    global_mae = df['absolute_error'].mean()
    global_mape = df['relative_error'].mean()

    print(f"MSE global: {global_mse:.4f}")
    print(f"RMSE global: {global_rmse:.4f}")
    print(f"MAE global: {global_mae:.4f}")
    print(f"MAPE global: {global_mape:.2f}%")

    return df, results

def plot_error_analysis(df, results, output_prefix='pricing-data/results/error_analysis'):
    """Visualise l'analyse de l'erreur"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Analyse de l\'erreur en fonction du prix', fontsize=16, fontweight='bold')

    # 1. Scatter plot: Prédictions vs Vraies valeurs
    ax = axes[0, 0]
    ax.scatter(df['true_price'], df['predicted_price'], alpha=0.5, s=10)
    min_val = min(df['true_price'].min(), df['predicted_price'].min())
    max_val = max(df['true_price'].max(), df['predicted_price'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction parfaite')
    ax.set_xlabel('Prix réel')
    ax.set_ylabel('Prix prédit')
    ax.set_title('Prédictions vs Réalité')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Erreur absolue vs Prix réel
    ax = axes[0, 1]
    ax.scatter(df['true_price'], df['absolute_error'], alpha=0.5, s=10, c=df['true_price'], cmap='viridis')
    ax.set_xlabel('Prix réel')
    ax.set_ylabel('Erreur absolue')
    ax.set_title('Erreur absolue en fonction du prix')
    ax.grid(True, alpha=0.3)

    # 3. Erreur relative vs Prix réel
    ax = axes[0, 2]
    ax.scatter(df['true_price'], df['relative_error'], alpha=0.5, s=10, c=df['true_price'], cmap='plasma')
    ax.set_xlabel('Prix réel')
    ax.set_ylabel('Erreur relative (%)')
    ax.set_title('Erreur relative en fonction du prix')
    ax.set_ylim(0, min(df['relative_error'].quantile(0.95), 100))  # Limiter à 95e percentile
    ax.grid(True, alpha=0.3)

    # 4. Distribution des erreurs par range
    ax = axes[1, 0]
    results_df = pd.DataFrame(results)
    x_pos = np.arange(len(results_df))
    ax.bar(x_pos, results_df['RMSE'], alpha=0.7, color='steelblue', label='RMSE')
    ax.bar(x_pos, results_df['MAE'], alpha=0.7, color='coral', label='MAE')
    ax.set_xlabel('Range de prix')
    ax.set_ylabel('Erreur')
    ax.set_title('RMSE et MAE par range de prix')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df['range'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 5. MAPE par range
    ax = axes[1, 1]
    ax.bar(x_pos, results_df['MAPE_%'], alpha=0.7, color='green')
    ax.set_xlabel('Range de prix')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Erreur relative (MAPE) par range')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df['range'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # 6. Distribution des prix dans chaque range
    ax = axes[1, 2]
    ax.bar(x_pos, results_df['count'], alpha=0.7, color='purple')
    ax.set_xlabel('Range de prix')
    ax.set_ylabel('Nombre d\'échantillons')
    ax.set_title('Distribution des échantillons par range')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df['range'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Graphique sauvegardé: {output_prefix}.png")

    # Graphique supplémentaire: Erreur vs Prix (binned)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Créer des bins de prix
    price_bins = np.linspace(0, df['true_price'].max(), 20)
    df['price_bin'] = pd.cut(df['true_price'], bins=price_bins)

    # Calculer les statistiques par bin
    binned_stats = df.groupby('price_bin', observed=True).agg({
        'absolute_error': ['mean', 'std', 'median'],
        'relative_error': 'mean',
        'true_price': 'mean'
    }).reset_index()

    binned_stats.columns = ['price_bin', 'mae_mean', 'mae_std', 'mae_median', 'mape', 'price_mean']

    ax.errorbar(binned_stats['price_mean'], binned_stats['mae_mean'],
                yerr=binned_stats['mae_std'], fmt='o-', linewidth=2,
                markersize=8, capsize=5, label='MAE ± std')
    ax.plot(binned_stats['price_mean'], binned_stats['mae_median'],
            's--', linewidth=2, markersize=6, label='MAE médiane', color='orange')

    ax.set_xlabel('Prix réel', fontsize=12)
    ax.set_ylabel('Erreur absolue', fontsize=12)
    ax.set_title('Évolution de l\'erreur en fonction du prix (avec bins)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_binned.png', dpi=300, bbox_inches='tight')
    print(f"✅ Graphique binné sauvegardé: {output_prefix}_binned.png")

def main():
    """Analyse principale"""

    experiments = [
        ('baseline_norm', 'pricing-data/results/baseline_norm_validation.csv'),
        ('simple_norm', 'pricing-data/results/simple_norm_validation.csv'),
        ('deep_norm', 'pricing-data/results/deep_norm_validation.csv'),
    ]

    all_results = []

    for exp_name, validation_file in experiments:
        if not Path(validation_file).exists():
            print(f"\n⚠️  Fichier manquant: {validation_file}")
            continue

        print(f"\n{'=' * 80}")
        print(f"EXPÉRIENCE: {exp_name}")
        print(f"{'=' * 80}")

        df, results = analyze_predictions(validation_file, 'pricing-data/valid.csv')

        # Ajouter le nom de l'expérience aux résultats
        for r in results:
            r['experiment'] = exp_name
        all_results.extend(results)

        # Créer les visualisations
        plot_error_analysis(df, results,
                          output_prefix=f'pricing-data/results/error_analysis_{exp_name}')

    # Sauvegarder les résultats dans un CSV
    results_df = pd.DataFrame(all_results)
    output_csv = 'pricing-data/results/error_by_price_range.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\n✅ Résultats sauvegardés: {output_csv}")

    # Comparaison entre expériences
    if len(all_results) > 0:
        print("\n" + "=" * 80)
        print("COMPARAISON ENTRE EXPÉRIENCES")
        print("=" * 80)
        pivot = results_df.pivot_table(
            index='range',
            columns='experiment',
            values='MAPE_%',
            aggfunc='first'
        )
        print("\nMAPE (%) par range et par expérience:")
        print(pivot.to_string())

if __name__ == '__main__':
    main()
