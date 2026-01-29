#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

experiments = [
    "baseline",
    "lr_high", 
    "no_momentum",
    "deep",
    "relu",
    "no_reg",
    "simple"
]

def get_final_error(error_file):
    """Extrait le MSE et RMSE finaux du fichier error CSV"""
    try:
        df = pd.read_csv(error_file, header=None, sep=' ')
        final_mse = df[0].iloc[-1]
        final_rmse = df[1].iloc[-1]
        return final_mse, final_rmse
    except Exception as e:
        print(f" Erreur pour {error_file}: {e}")
        return None, None

results = []

print("COMPARAISON: AVEC vs SANS NORMALISATION")

for exp in experiments:
    error_file = f"pricing-data/results/{exp}_error.csv"
    mse_no_norm, rmse_no_norm = get_final_error(error_file)
    
    error_file_norm = f"pricing-data/results/{exp}_norm_error.csv"
    mse_norm, rmse_norm = get_final_error(error_file_norm)
    
    results.append({
        'Experiment': exp,
        'MSE (No Norm)': mse_no_norm,
        'RMSE (No Norm)': rmse_no_norm,
        'MSE (Norm)': mse_norm,
        'RMSE (Norm)': rmse_norm,
        'Improvement (%)': ((mse_no_norm - mse_norm) / mse_no_norm * 100) if (mse_no_norm and mse_norm) else None
    })
    
    print(f"\n{exp.upper()}")
    if mse_no_norm and mse_norm:
        print(f"  Sans normalisation: MSE={mse_no_norm:.2f}, RMSE={rmse_no_norm:.2f}")
        print(f"  Avec normalisation: MSE={mse_norm:.2f}, RMSE={rmse_norm:.2f}")
        improvement = (mse_no_norm - mse_norm) / mse_no_norm * 100
        print(f"  Amélioration: {improvement:+.1f}%")
    else:
        print(f" Données manquantes")

df = pd.DataFrame(results)

df.to_csv('pricing-data/results/normalization_comparison.csv', index=False)
print("\n Résultats sauvegardés dans pricing-data/results/normalization_comparison.csv")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0, 0]
x = np.arange(len(experiments))
width = 0.35
ax.bar(x - width/2, df['MSE (No Norm)'], width, label='Sans normalisation', alpha=0.8, color='#FF6B6B')
ax.bar(x + width/2, df['MSE (Norm)'], width, label='Avec normalisation', alpha=0.8, color='#4ECDC4')
ax.set_ylabel('MSE (Validation)', fontsize=12, fontweight='bold')
ax.set_title('MSE sur ensemble de validation', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(experiments, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

ax = axes[0, 1]
ax.bar(x - width/2, df['RMSE (No Norm)'], width, label='Sans normalisation', alpha=0.8, color='#FF6B6B')
ax.bar(x + width/2, df['RMSE (Norm)'], width, label='Avec normalisation', alpha=0.8, color='#4ECDC4')
ax.set_ylabel('RMSE (Validation)', fontsize=12, fontweight='bold')
ax.set_title('RMSE sur ensemble de validation', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(experiments, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

ax = axes[1, 0]
improvements = df['Improvement (%)'].fillna(0)
colors = ['#51CF66' if imp > 0 else '#FF6B6B' for imp in improvements]
bars = ax.bar(x, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Amélioration (%)', fontsize=12, fontweight='bold')
ax.set_title('Amélioration apportée par la normalisation', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(experiments, rotation=45, ha='right')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.grid(axis='y', alpha=0.3, linestyle='--')

for i, bar in enumerate(bars):
    height = bar.get_height()
    if not np.isnan(height):
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.1f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold', fontsize=9)

ax = axes[1, 1]
ax.axis('off')

table_data = []
for _, row in df.iterrows():
    table_data.append([
        row['Experiment'],
        f"{row['MSE (No Norm)']:.1f}" if row['MSE (No Norm)'] else 'N/A',
        f"{row['MSE (Norm)']:.1f}" if row['MSE (Norm)'] else 'N/A',
        f"{row['Improvement (%)']:+.1f}%" if row['Improvement (%)'] else 'N/A'
    ])

table = ax.table(
    cellText=table_data,
    colLabels=['Expérience', 'MSE\n(Sans norm)', 'MSE\n(Avec norm)', 'Amélioration'],
    cellLoc='center',
    loc='center',
    colWidths=[0.25, 0.25, 0.25, 0.25]
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

for i in range(len(table_data) + 1):
    for j in range(4):
        cell = table[(i, j)]
        if i == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(weight='bold', color='white', fontsize=11)
        else:
            cell.set_edgecolor('#CCCCCC')
            if j == 3 and table_data[i-1][3] != 'N/A':
                val = float(table_data[i-1][3].strip('%+'))
                if val > 0:
                    cell.set_facecolor('#C8E6C9')
                    cell.set_text_props(weight='bold', color='#1B5E20')
                else:
                    cell.set_facecolor('#FFCDD2')
                    cell.set_text_props(weight='bold', color='#B71C1C')

plt.suptitle('Impact de la normalisation des données sur les performances', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('pricing-data/results/normalization_impact.png', dpi=150, bbox_inches='tight')
print("Graphiques sauvegardés dans pricing-data/results/normalization_impact.png")

best_no_norm = df.loc[df['MSE (No Norm)'].idxmin()]
best_norm = df.loc[df['MSE (Norm)'].idxmin()]

print("\n" + "=" * 80)
print("MEILLEURS RÉSEAUX")
print(f"\nSans normalisation:")
print(f"  {best_no_norm['Experiment']}: MSE = {best_no_norm['MSE (No Norm)']:.2f}, RMSE = {best_no_norm['RMSE (No Norm)']:.2f}")

print(f"\nAvec normalisation:")
print(f"  {best_norm['Experiment']}: MSE = {best_norm['MSE (Norm)']:.2f}, RMSE = {best_norm['RMSE (Norm)']:.2f}")

overall_improvement = ((best_no_norm['MSE (No Norm)'] - best_norm['MSE (Norm)']) / best_no_norm['MSE (No Norm)'] * 100)
print(f"\nGain global avec le meilleur réseau: {overall_improvement:.1f}%")
print("=" * 80)

print("STATISTIQUES GÉNÉRALES")
avg_improvement = df['Improvement (%)'].mean()
print(f"Amélioration moyenne: {avg_improvement:+.1f}%")
print(f"Meilleure amélioration: {df['Improvement (%)'].max():+.1f}% ({df.loc[df['Improvement (%)'].idxmax(), 'Experiment']})")
print(f"Pire amélioration: {df['Improvement (%)'].min():+.1f}% ({df.loc[df['Improvement (%)'].idxmin(), 'Experiment']})")
