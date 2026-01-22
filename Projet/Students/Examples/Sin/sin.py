import math
import random

def generate_data(filename, n_samples):
    with open(filename, 'w') as f:
        for _ in range(n_samples):
            x = random.uniform(-math.pi, math.pi)
            y = math.sin(x)
            f.write(f"{x},{y}\n")

if __name__ == "__main__":
    generate_data("sin_train.csv", 2000)
    generate_data("sin_valid.csv", 500)
    print("Files sin_train.csv and sin_valid.csv generated.")
