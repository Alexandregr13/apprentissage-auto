package fr.ensimag.deep.layers;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import fr.ensimag.deep.activators.Identity;
import fr.ensimag.deep.activators.Relu;

public class LayerTest {

    @Test
    public void testForwardPropagation() {
        // Setup: Input size 2, Output size 1
        SimpleMatrix weights = new SimpleMatrix(new double[][] {
            {2.0}, // Weight for input 0
            {3.0}  // Weight for input 1
        });
        
        SimpleMatrix bias = new SimpleMatrix(new double[][] {
            {5.0} // Bias for neuron 0
        });
        
        StandardLayer layer = new StandardLayer(weights, bias, new Identity());
        
        // Input: 2 features, 1 example
        SimpleMatrix input = new SimpleMatrix(new double[][] {
            {10.0},
            {100.0}
        });
        
        layer.propagate(input);
        SimpleMatrix output = layer.getActivation();
        
        // Expected: (2.0 * 10.0) + (3.0 * 100.0) + 5.0 = 20 + 300 + 5 = 325
        assertEquals(325.0, output.get(0, 0), 1e-6);
    }

    @Test
    public void testForwardPropagationBatch() {
        // Setup: Input size 2, Output size 1
        SimpleMatrix weights = new SimpleMatrix(new double[][] {
            {1.0}, 
            {-1.0}
        });
        
        SimpleMatrix bias = new SimpleMatrix(new double[][] {
            {0.0} 
        });
        
        StandardLayer layer = new StandardLayer(weights, bias, new Relu());
        
        // Input: 2 features, 2 examples
        SimpleMatrix input = new SimpleMatrix(new double[][] {
            {10.0, 50.0},
            {5.0,  60.0}
        });
        
        layer.propagate(input);
        SimpleMatrix output = layer.getActivation();
        
        // 1*10 + (-1)*5 + 0 = 5. Relu(5) = 5.
        assertEquals(5.0, output.get(0, 0), 1e-6);
        
        // 1*50 + (-1)*60 + 0 = -10. Relu(-10) = 0.
        assertEquals(0.0, output.get(0, 1), 1e-6);
    }

    @Test
    public void testBackpropagationAndMomentum() {
        // Setup simple layer: 1 input -> 1 output
        SimpleMatrix weights = new SimpleMatrix(new double[][] {{ 0.5 }});
        SimpleMatrix bias = new SimpleMatrix(new double[][] {{ 0.0 }});
        StandardLayer layer = new StandardLayer(weights, bias, new Identity()); // Identity for simple derivative=1
        
        layer.setLearningRate(0.1);
        layer.setMomentumCoefficient(0.9); // With momentum
        
        // Forward
        // Input = 2.0
        SimpleMatrix input = new SimpleMatrix(new double[][] {{ 2.0 }});
        layer.propagate(input);
        
        // Output = 0.5 * 2.0 = 1.0
        // WeightedInput = 1.0 (since bias 0)
        
        // Backprop
        // Assume Error from next layer (or cost derivative) is 1.0
        SimpleMatrix nextError = new SimpleMatrix(new double[][] {{ 1.0 }});
        layer.backpropagate(nextError);
        
        // Local error = phi'(z) * nextError = 1.0 * 1.0 = 1.0
        
        // Update
        // Gradient = Activation_prev * localError = 2.0 * 1.0 = 2.0
        // Velocity (new) = (0.9 * 0) - (2.0 * 0.1) = -0.2
        // Weights (new) = 0.5 + (-0.2) = 0.3
        
        layer.updateParameters();
        
        assertEquals(0.3, layer.getWeights().get(0, 0), 1e-6, "Weights should update with momentum");
    }

    @Test
    public void testL2Regularization() {
         // Setup simple layer: 1 input -> 1 output
         SimpleMatrix weights = new SimpleMatrix(new double[][] {{ 10.0 }}); // Large weight
         SimpleMatrix bias = new SimpleMatrix(new double[][] {{ 0.0 }});
         StandardLayer layer = new StandardLayer(weights, bias, new Identity());
         
         layer.setLearningRate(0.1);
         layer.setL2RegularizationCoefficient(0.5); // Large lambda
         
         // Forward (Input 0 to avoid gradient from data)
         SimpleMatrix input = new SimpleMatrix(new double[][] {{ 0.0 }});
         layer.propagate(input);
         
         // Backprop (Error 0)
         SimpleMatrix nextError = new SimpleMatrix(new double[][] {{ 0.0 }});
         layer.backpropagate(nextError);
         
         // Update
         // Gradient Data = 0 * 0 = 0
         // Gradient L2 = (lambda/n) * w = (0.5/1) * 10 = 5.0
         // Total Gradient = 5.0
         // Weight Update = - (LearningRate * Gradient) = - (0.1 * 5.0) = -0.5
         // New Weight = 10.0 - 0.5 = 9.5
         // NOTE: Implementation uses: velocity = mom*v - lr*grad
         // v = 0 - 0.1 * 5.0 = -0.5
         // w = 10.0 + (-0.5) = 9.5
         
         layer.updateParameters();
         
         assertEquals(9.5, layer.getWeights().get(0, 0), 1e-6, "Weights should shrink due to L2");
    }
}
