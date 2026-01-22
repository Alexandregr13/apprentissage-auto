package fr.ensimag.deep.trainer.operators;

import org.ejml.simple.SimpleMatrix;

import fr.ensimag.deep.trainer.costFunction.ICostFunction;

public interface ICommand {

    /**
     * 
     * @param expectedOutput
     * @param output
     * @param cost
     */
    public void compute(SimpleMatrix expectedOutput, SimpleMatrix output, SimpleMatrix cost);
}
