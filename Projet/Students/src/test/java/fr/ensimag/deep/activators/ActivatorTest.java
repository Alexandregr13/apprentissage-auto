package fr.ensimag.deep.activators;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

public class ActivatorTest {

    private static final double DELTA = 1e-6;

    @Test
    public void testSigmoid() {
        IActivator sigmoid = new Sigmoid();
        assertEquals(0.5, sigmoid.phi(0), DELTA, "Sigmoid(0) should be 0.5");
        assertTrue(sigmoid.phi(10) > 0.99, "Sigmoid(10) should be close to 1");
        assertTrue(sigmoid.phi(-10) < 0.01, "Sigmoid(-10) should be close to 0");

        // phiPrime(0) = phi(0)*(1-phi(0)) = 0.5 * 0.5 = 0.25
        assertEquals(0.25, sigmoid.phiPrime(0), DELTA, "Sigmoid'(0) should be 0.25");
    }

    @Test
    public void testTanh() {
        IActivator tanh = new Tanh();
        assertEquals(0.0, tanh.phi(0), DELTA, "Tanh(0) should be 0");
        
        // phiPrime(0) = 1 - phi(0)^2 = 1
        assertEquals(1.0, tanh.phiPrime(0), DELTA, "Tanh'(0) should be 1");
    }

    @Test
    public void testRelu() {
        IActivator relu = new Relu();

        // phi(positive) = identity
        assertEquals(5.0, relu.phi(5.0), DELTA);
               
        // phi(negative) = 0
        assertEquals(0.0, relu.phi(-5.0), DELTA);

        // phiPrime(positive) = 1
        assertEquals(1.0, relu.phiPrime(5.0), DELTA);
        
        // phiPrime(negative) = 0
        assertEquals(0.0, relu.phiPrime(-5.0), DELTA);
    }
}
