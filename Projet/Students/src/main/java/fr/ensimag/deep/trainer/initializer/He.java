package fr.ensimag.deep.trainer.initializer;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.ejml.simple.SimpleMatrix;

import fr.ensimag.deep.NeuralNetwork;
import fr.ensimag.deep.layers.AbstractLayer;

/**
 * Random initialization of weights
 */
public class He implements INewtorkInitializer {
    private NormalDistribution distribution = null;

    public He()
    {
        this.distribution = new NormalDistribution();
    }

    /** 
     * Random initialization of weights and bias of the layers of the network
     * Weights are drawn from a Gaussian distribution with standard deviation 
     * of sqrt(2/n)) where n is the size of the current layer
     * distribution.sample() draws a number from a gaussian distribution centered on 0
     * with a standard deviation of 1
     * @param neuralNetwork
     */
    public void init(NeuralNetwork neuralNetwork)
     { 
        for (AbstractLayer layer : neuralNetwork.getLayers()) {
            int n = layer.getPreviousLayerSize(); // le commentaire dis current mais c'est prec ?
            double stdDev = Math.sqrt(2.0 / n);
            
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
