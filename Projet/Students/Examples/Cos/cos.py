import math
import random

def generate_data(filename, n_samples):
    with open(filename, 'w') as f:
        for _ in range(n_samples):
            x = random.uniform(-math.pi, math.pi)
            y = math.cos(x)
            
            # Write "Input,Output" for CSVReader
            f.write(f"{x},{y}\n")

if __name__ == "__main__":
    generate_data("cos_train.csv", 2000)
    generate_data("cos_valid.csv", 500)
    print("Files cos_train.csv and cos_valid.csv generated.")
