package fr.ensimag.deep.trainer.costFunction;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

public class CostFunctionTest {

    private static final double DELTA = 1e-6;

    @Test
    public void testQuadraticCostFunction() {
        ICostFunction costFunction = new QuadraticCostFunction();

        // Test derivative (what we really use in backprop)
        // derivApply(y, expected_y) = (y - expected_y)
        
        // Expected: 0.8 - 1.0 = -0.2
        assertEquals(-0.2, costFunction.derivApply(0.8, 1.0), DELTA, "Deriv index incorrect");
        
        // Expected: 0.3 - 0.0 = 0.3
        assertEquals(0.3, costFunction.derivApply(0.3, 0.0), DELTA, "Deriv index incorrect");
        
        // Test cost (1/2 * (y - t)^2)
        // 0.5 * (-0.2)^2 = 0.5 * 0.04 = 0.02
        assertEquals(0.02, costFunction.apply(0.8, 1.0), DELTA, "Cost incorrect");
    }
}
