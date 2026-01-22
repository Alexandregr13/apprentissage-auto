import matplotlib.pyplot as plt
import csv
import math
import sys
import os

def plot_file(filename, func_name=None):
    x_pred = []
    y_pred = []

    with open(filename, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            if len(row) >= 2:
                x_pred.append(float(row[0]))
                y_pred.append(float(row[1]))
                
    if not x_pred:
        print("No valid data found in file.")
        return

    plt.figure(figsize=(10, 6))

    if func_name in ['sin', 'cos']:
        x_true = sorted(x_pred)
        if func_name == 'sin':
            y_true = [math.sin(val) for val in x_true]
            label_true = 'True Sin(x)'
        elif func_name == 'cos':
            y_true = [math.cos(val) for val in x_true]
            label_true = 'True Cos(x)'
        
        plt.plot(x_true, y_true, color='green', label=label_true, linewidth=2)

    plt.scatter(x_pred, y_pred, color='red', s=10, label='Predicted', alpha=0.5)
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title(f'Neural Network Output: {os.path.basename(filename)}')
    plt.legend()
    plt.grid(True)

    output_png = os.path.splitext(filename)[0] + '.png'
    plt.savefig(output_png)
    print(f"Plot saved as {output_png}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot.py <csv_file> [func_name]")
        print("Example: python plot.py sin_final_validation.csv sin")
    else:
        file_path = sys.argv[1]
        func = sys.argv[2] if len(sys.argv) > 2 else None
        if func and func not in ['sin', 'cos']:
            print("Warning: function name should be 'sin' or 'cos' for comparison curve. Plotting only predictions.")
        plot_file(file_path, func)
