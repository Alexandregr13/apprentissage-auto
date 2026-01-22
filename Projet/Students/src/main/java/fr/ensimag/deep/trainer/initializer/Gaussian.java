package fr.ensimag.deep.trainer.initializer;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.ejml.simple.SimpleMatrix;

import fr.ensimag.deep.NeuralNetwork;
import fr.ensimag.deep.layers.AbstractLayer;

/**
 * Random initialization of weights
 */
public class Gaussian implements INewtorkInitializer {
    private NormalDistribution distribution = null;

    public Gaussian()
    {
        this.distribution = new NormalDistribution();
    }

    /** 
     * Random initialization of weights and bias of the layers of the network
     * Weights are drawn from a Gaussian distribution with standard deviation 
     * of 1/sqrt(n) where n is the size of the previous layer
     * distribution.sample() draws a number from a gaussian distribution centered on 0
     * with a standard deviation of 1
     * @param neuralNetwork
     */
    public void init(NeuralNetwork neuralNetwork)
     {
        for (AbstractLayer layer : neuralNetwork.getLayers()) {
            int n = layer.getPreviousLayerSize();
            double stdDev = 1.0 / Math.sqrt(n);
            
            SimpleMatrix weights = layer.getWeights();
            for (int r = 0; r < weights.numRows(); r++) {
                for (int c = 0; c < weights.numCols(); c++) {
                    weights.set(r, c, distribution.sample() * stdDev);
                }
            }
            
            SimpleMatrix bias = layer.getBias();
            if (bias != null) {
                for (int r = 0; r < bias.numRows(); r++) {
                    bias.set(r, 0, 0.0);
                }
            }
        }
     }
}
