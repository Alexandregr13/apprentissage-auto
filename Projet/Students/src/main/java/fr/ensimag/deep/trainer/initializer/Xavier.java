package fr.ensimag.deep.trainer.initializer;

import org.ejml.simple.SimpleMatrix;

import fr.ensimag.deep.NeuralNetwork;
import fr.ensimag.deep.layers.AbstractLayer;

/**
 * Random initialization of weights
 */
public class Xavier implements INewtorkInitializer {
    private java.util.Random random = new java.util.Random();

    public Xavier()
    {
    }

   /** 
     * Random initialization of weights and bias of the layers of the network
     * Weights are drawn from a Uniform distribution on [-sqrt(6/(n+m)), sqrt(6/(n+m))]
     * where n is the size of the previous layer and m the size of the current layer
     * @param neuralNetwork
     */
    public void init(NeuralNetwork neuralNetwork)
     {
        for (AbstractLayer layer : neuralNetwork.getLayers()) {
            int n = layer.getPreviousLayerSize();
            int m = layer.getLayerSize();
            double limit = Math.sqrt(6.0 / (n + m));
            
            SimpleMatrix weights = layer.getWeights();
            for (int r = 0; r < weights.numRows(); r++) {
                for (int c = 0; c < weights.numCols(); c++) {
                    // Uniform distribution between -limit and limit
                    double val = (random.nextDouble() * 2.0 * limit) - limit;
                    weights.set(r, c, val);
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
