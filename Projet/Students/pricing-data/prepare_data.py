#!/usr/bin/env python3
import random

with open('pricing-data/pricing-data-inputs.csv', 'r') as f:
    inputs = [line.strip() for line in f if line.strip()]

with open('pricing-data/pricing-data-outputs.csv', 'r') as f:
    outputs = [line.strip() for line in f if line.strip()]

data = [f"{inp.rstrip(',')},{out}" for inp, out in zip(inputs, outputs)]
random.seed(42)
random.shuffle(data)

n_train = int(len(data) * 0.70)
n_valid = int(len(data) * 0.15)

with open('pricing-data/train.csv', 'w') as f:
    f.write('\n'.join(data[:n_train]) + '\n')

with open('pricing-data/valid.csv', 'w') as f:
    f.write('\n'.join(data[n_train:n_train+n_valid]) + '\n')

with open('pricing-data/test.csv', 'w') as f:
    f.write('\n'.join(data[n_train+n_valid:]) + '\n')

print(f"Done: {n_train} train, {n_valid} valid, {len(data)-n_train-n_valid} test")

print("Vous pouvez maintenant lancer: ./pricing-data/RUN_ALL.sh")
