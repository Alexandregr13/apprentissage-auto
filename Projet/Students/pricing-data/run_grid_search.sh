#!/bin/bash
# Grid search automatique

echo '=== [1/56] grid_tiny_lr0.01_m0.9_l20.0001_tanh_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_tiny_lr0.01_m0.9_l20.0001_tanh_bs16.json"

echo '=== [2/56] grid_tiny_lr0.01_m0.9_l20.0001_tanh_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_tiny_lr0.01_m0.9_l20.0001_tanh_bs32.json"

echo '=== [3/56] grid_tiny_lr0.01_m0.9_l20.001_tanh_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_tiny_lr0.01_m0.9_l20.001_tanh_bs16.json"

echo '=== [4/56] grid_tiny_lr0.01_m0.9_l20.001_tanh_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_tiny_lr0.01_m0.9_l20.001_tanh_bs32.json"

echo '=== [5/56] grid_tiny_lr0.01_m0.9_l20.0001_relu_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_tiny_lr0.01_m0.9_l20.0001_relu_bs16.json"

echo '=== [6/56] grid_tiny_lr0.01_m0.9_l20.0001_relu_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_tiny_lr0.01_m0.9_l20.0001_relu_bs32.json"

echo '=== [7/56] grid_tiny_lr0.01_m0.9_l20.001_relu_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_tiny_lr0.01_m0.9_l20.001_relu_bs16.json"

echo '=== [8/56] grid_tiny_lr0.01_m0.9_l20.001_relu_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_tiny_lr0.01_m0.9_l20.001_relu_bs32.json"

echo '=== [9/56] grid_simple_lr0.01_m0.9_l20.0001_tanh_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_simple_lr0.01_m0.9_l20.0001_tanh_bs16.json"

echo '=== [10/56] grid_simple_lr0.01_m0.9_l20.0001_tanh_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_simple_lr0.01_m0.9_l20.0001_tanh_bs32.json"

echo '=== [11/56] grid_simple_lr0.01_m0.9_l20.001_tanh_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_simple_lr0.01_m0.9_l20.001_tanh_bs16.json"

echo '=== [12/56] grid_simple_lr0.01_m0.9_l20.001_tanh_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_simple_lr0.01_m0.9_l20.001_tanh_bs32.json"

echo '=== [13/56] grid_simple_lr0.01_m0.9_l20.0001_relu_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_simple_lr0.01_m0.9_l20.0001_relu_bs16.json"

echo '=== [14/56] grid_simple_lr0.01_m0.9_l20.0001_relu_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_simple_lr0.01_m0.9_l20.0001_relu_bs32.json"

echo '=== [15/56] grid_simple_lr0.01_m0.9_l20.001_relu_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_simple_lr0.01_m0.9_l20.001_relu_bs16.json"

echo '=== [16/56] grid_simple_lr0.01_m0.9_l20.001_relu_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_simple_lr0.01_m0.9_l20.001_relu_bs32.json"

echo '=== [17/56] grid_simple15_lr0.01_m0.9_l20.0001_tanh_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_simple15_lr0.01_m0.9_l20.0001_tanh_bs16.json"

echo '=== [18/56] grid_simple15_lr0.01_m0.9_l20.0001_tanh_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_simple15_lr0.01_m0.9_l20.0001_tanh_bs32.json"

echo '=== [19/56] grid_simple15_lr0.01_m0.9_l20.001_tanh_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_simple15_lr0.01_m0.9_l20.001_tanh_bs16.json"

echo '=== [20/56] grid_simple15_lr0.01_m0.9_l20.001_tanh_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_simple15_lr0.01_m0.9_l20.001_tanh_bs32.json"

echo '=== [21/56] grid_simple15_lr0.01_m0.9_l20.0001_relu_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_simple15_lr0.01_m0.9_l20.0001_relu_bs16.json"

echo '=== [22/56] grid_simple15_lr0.01_m0.9_l20.0001_relu_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_simple15_lr0.01_m0.9_l20.0001_relu_bs32.json"

echo '=== [23/56] grid_simple15_lr0.01_m0.9_l20.001_relu_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_simple15_lr0.01_m0.9_l20.001_relu_bs16.json"

echo '=== [24/56] grid_simple15_lr0.01_m0.9_l20.001_relu_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_simple15_lr0.01_m0.9_l20.001_relu_bs32.json"

echo '=== [25/56] grid_standard_lr0.01_m0.9_l20.0001_tanh_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_standard_lr0.01_m0.9_l20.0001_tanh_bs16.json"

echo '=== [26/56] grid_standard_lr0.01_m0.9_l20.0001_tanh_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_standard_lr0.01_m0.9_l20.0001_tanh_bs32.json"

echo '=== [27/56] grid_standard_lr0.01_m0.9_l20.001_tanh_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_standard_lr0.01_m0.9_l20.001_tanh_bs16.json"

echo '=== [28/56] grid_standard_lr0.01_m0.9_l20.001_tanh_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_standard_lr0.01_m0.9_l20.001_tanh_bs32.json"

echo '=== [29/56] grid_standard_lr0.01_m0.9_l20.0001_relu_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_standard_lr0.01_m0.9_l20.0001_relu_bs16.json"

echo '=== [30/56] grid_standard_lr0.01_m0.9_l20.0001_relu_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_standard_lr0.01_m0.9_l20.0001_relu_bs32.json"

echo '=== [31/56] grid_standard_lr0.01_m0.9_l20.001_relu_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_standard_lr0.01_m0.9_l20.001_relu_bs16.json"

echo '=== [32/56] grid_standard_lr0.01_m0.9_l20.001_relu_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_standard_lr0.01_m0.9_l20.001_relu_bs32.json"

echo '=== [33/56] grid_medium_lr0.01_m0.9_l20.0001_tanh_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_medium_lr0.01_m0.9_l20.0001_tanh_bs16.json"

echo '=== [34/56] grid_medium_lr0.01_m0.9_l20.0001_tanh_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_medium_lr0.01_m0.9_l20.0001_tanh_bs32.json"

echo '=== [35/56] grid_medium_lr0.01_m0.9_l20.001_tanh_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_medium_lr0.01_m0.9_l20.001_tanh_bs16.json"

echo '=== [36/56] grid_medium_lr0.01_m0.9_l20.001_tanh_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_medium_lr0.01_m0.9_l20.001_tanh_bs32.json"

echo '=== [37/56] grid_medium_lr0.01_m0.9_l20.0001_relu_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_medium_lr0.01_m0.9_l20.0001_relu_bs16.json"

echo '=== [38/56] grid_medium_lr0.01_m0.9_l20.0001_relu_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_medium_lr0.01_m0.9_l20.0001_relu_bs32.json"

echo '=== [39/56] grid_medium_lr0.01_m0.9_l20.001_relu_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_medium_lr0.01_m0.9_l20.001_relu_bs16.json"

echo '=== [40/56] grid_medium_lr0.01_m0.9_l20.001_relu_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_medium_lr0.01_m0.9_l20.001_relu_bs32.json"

echo '=== [41/56] grid_deep_lr0.01_m0.9_l20.0001_tanh_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_deep_lr0.01_m0.9_l20.0001_tanh_bs16.json"

echo '=== [42/56] grid_deep_lr0.01_m0.9_l20.0001_tanh_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_deep_lr0.01_m0.9_l20.0001_tanh_bs32.json"

echo '=== [43/56] grid_deep_lr0.01_m0.9_l20.001_tanh_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_deep_lr0.01_m0.9_l20.001_tanh_bs16.json"

echo '=== [44/56] grid_deep_lr0.01_m0.9_l20.001_tanh_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_deep_lr0.01_m0.9_l20.001_tanh_bs32.json"

echo '=== [45/56] grid_deep_lr0.01_m0.9_l20.0001_relu_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_deep_lr0.01_m0.9_l20.0001_relu_bs16.json"

echo '=== [46/56] grid_deep_lr0.01_m0.9_l20.0001_relu_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_deep_lr0.01_m0.9_l20.0001_relu_bs32.json"

echo '=== [47/56] grid_deep_lr0.01_m0.9_l20.001_relu_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_deep_lr0.01_m0.9_l20.001_relu_bs16.json"

echo '=== [48/56] grid_deep_lr0.01_m0.9_l20.001_relu_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_deep_lr0.01_m0.9_l20.001_relu_bs32.json"

echo '=== [49/56] grid_very_deep_lr0.01_m0.9_l20.0001_tanh_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_very_deep_lr0.01_m0.9_l20.0001_tanh_bs16.json"

echo '=== [50/56] grid_very_deep_lr0.01_m0.9_l20.0001_tanh_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_very_deep_lr0.01_m0.9_l20.0001_tanh_bs32.json"

echo '=== [51/56] grid_very_deep_lr0.01_m0.9_l20.001_tanh_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_very_deep_lr0.01_m0.9_l20.001_tanh_bs16.json"

echo '=== [52/56] grid_very_deep_lr0.01_m0.9_l20.001_tanh_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_very_deep_lr0.01_m0.9_l20.001_tanh_bs32.json"

echo '=== [53/56] grid_very_deep_lr0.01_m0.9_l20.0001_relu_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_very_deep_lr0.01_m0.9_l20.0001_relu_bs16.json"

echo '=== [54/56] grid_very_deep_lr0.01_m0.9_l20.0001_relu_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_very_deep_lr0.01_m0.9_l20.0001_relu_bs32.json"

echo '=== [55/56] grid_very_deep_lr0.01_m0.9_l20.001_relu_bs16 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_very_deep_lr0.01_m0.9_l20.001_relu_bs16.json"

echo '=== [56/56] grid_very_deep_lr0.01_m0.9_l20.001_relu_bs32 ==='
mvn exec:java -Dexec.mainClass="fr.ensimag.deep.trainingConsole.Main" \
  -Dexec.args="-x pricing-data/experiments/expe_grid_very_deep_lr0.01_m0.9_l20.001_relu_bs32.json"

