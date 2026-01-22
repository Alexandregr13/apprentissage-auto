package fr.ensimag.deep.layers;

import org.ejml.simple.SimpleMatrix;

import fr.ensimag.deep.activators.IActivator;

public class StandardLayer extends AbstractLayer{

    private SimpleMatrix weights;
    private SimpleMatrix bias;
    private SimpleMatrix weightedInput;
    private SimpleMatrix activation;

    private SimpleMatrix localError;
    private SimpleMatrix previousActivation;
    private SimpleMatrix weightedError;

    private SimpleMatrix velocity;
    private SimpleMatrix biasVelocity;
    private double momentumCoefficient = 0.9;

    private double l2RegularizationCoefficient = 0.0; // lambda


    public StandardLayer(SimpleMatrix initialWeights, SimpleMatrix initialBias, IActivator activator)
    {
        super(initialBias.getNumRows(), initialWeights.getNumRows(), activator);
        this.weights = initialWeights;
        this.bias = initialBias;
    }

    @Override
    public void propagate(SimpleMatrix input) {
        //throw new UnsupportedOperationException("not yet implemented");

        //this.weightedInput = weights.transpose().mult(input).plus(bias);
        // Ca marche pas le plus car pb de dimensionnement des matrices

        int numExamples = input.getNumCols();
        int numNeurons = bias.getNumRows();

        this.previousActivation = input.copy();


        if (this.weightedInput == null || weightedInput.getNumCols() != numExamples){
            this.weightedInput = new SimpleMatrix(bias.getNumRows(), numExamples);
        }

        SimpleMatrix temp = weights.transpose().mult(input);

        for (int j = 0; j < numExamples; j++) {
            for (int i = 0; i < numNeurons; i++) {
                double value = temp.get(i,j) + bias.get(i,0);
                weightedInput.set(i, j, value);
            }
        }


        if (this.activation == null || activation.getNumCols() != numExamples) {
            this.activation = new SimpleMatrix(bias.getNumRows(), numExamples);
        }

        for (int i = 0; i < activation.numRows(); i++) {
            for (int j = 0; j < activation.getNumCols(); j++) {
                double value = weightedInput.get(i,j);
                activation.set(i, j, activator.phi(value));
            }
        }
    }

    @Override
    public SimpleMatrix getActivation() {
        //throw new UnsupportedOperationException("not yet implemented");
        return activation;
    }

    @Override
    public SimpleMatrix getWeightedError() {
        //throw new UnsupportedOperationException("not yet implemented");
        return weightedError;
    }    

    @Override
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        // to be completed
    }

    public void setL2RegularizationCoefficient(double lambda) {
        this.l2RegularizationCoefficient = lambda;
    }

    public double getL2RegularizationCoefficient() {
        return this.l2RegularizationCoefficient;
    }

    //seance 2 + 3 (momentum) +4 (regularization)
    @Override
    public void updateParameters() {

        int numExamples = localError.numCols(); //nb erreur dans le batch

        SimpleMatrix weightsGradient = previousActivation.mult(localError.transpose());
        weightsGradient = weightsGradient.scale(1.0 / numExamples);

        if (l2RegularizationCoefficient > 0) {
            SimpleMatrix l2Penalty = weights.scale(l2RegularizationCoefficient / numExamples);
            weightsGradient = weightsGradient.plus(l2Penalty);
        }
        SimpleMatrix biasGradient = new SimpleMatrix(bias.numRows(), 1);
        for (int i = 0; i < localError.numRows(); i++) {
            double sum = 0;
            for (int j = 0; j < numExamples; j++) {
                sum += localError.get(i, j);
            }
            biasGradient.set(i, 0, sum/numExamples);
        }


        if(this.velocity == null) {
            this.velocity = new SimpleMatrix(weights.numRows(), weights.numCols());
        }
        if (this.biasVelocity == null) {
            this.biasVelocity = new SimpleMatrix(bias.numRows(), bias.numCols());
        }

        this.velocity = velocity.scale(momentumCoefficient).minus(weightsGradient.scale(learningRate));
        this.biasVelocity = biasVelocity.scale(momentumCoefficient).minus(biasGradient.scale(learningRate));

        //SimpleMatrix weightUpdate = weightsGradient.scale(learningRate);
        //this.weights = this.weights.minus(weightUpdate); // On soustrait pour aller dans la direction qui réduit l'erreur

        //SimpleMatrix biasUpdate = biasGradient.scale(learningRate);
        //this.bias = this.bias.minus(biasUpdate);

        this.weights = this.weights.plus(velocity);
        this.bias = this.bias.plus(biasVelocity);

    }

    //seance 2
    @Override
    public void backpropagate(SimpleMatrix upperWeightedError) {
        //throw new UnsupportedOperationException("not yet implemented");

        int numNeurons = weightedInput.numRows();
        int numExamples = weightedInput.numCols();

        if (this.localError == null || localError.getNumCols() != numExamples) {
            this.localError = new SimpleMatrix(numNeurons, numExamples);
        }

        for (int i = 0; i < numNeurons; i++) {
            for (int j = 0; j < numExamples; j++) {
                double phiPrimeValue = activator.phiPrime(weightedInput.get(i, j));
                double errorValue = upperWeightedError.get(i, j);
                localError.set(i, j, phiPrimeValue * errorValue);
            }
        }
        if (this.weightedError == null || weightedError.getNumCols() != numExamples) {
            this.weightedError = new SimpleMatrix(weights.numRows(), numExamples);
        }

        SimpleMatrix temp = weights.mult(localError);
        this.weightedError = temp;
    }
                

    @Override
    public SimpleMatrix getWeights() {
        //throw new UnsupportedOperationException("not yet implemented");
        return weights;
    }

    @Override
    public SimpleMatrix getBias() {
        //throw new UnsupportedOperationException("not yet implemented");
        return bias;
    }

    @Override
    public void setWeights(SimpleMatrix weights) {
        //throw new UnsupportedOperationException("not yet implemented");
        this.weights = weights;
    }

    @Override
    public void setBias(SimpleMatrix bias) {
        //throw new UnsupportedOperationException("not yet implemented");
        this.bias = bias;
    }

    public void setMomentumCoefficient(double momentum) {
        this.momentumCoefficient = momentum;
    }

    public double getMomentumCoefficient() {
        return this.momentumCoefficient;
    }

    public void resetVelocity() {
        if(this.velocity != null) {
            this.velocity = new SimpleMatrix(weights.numRows(), weights.numCols());
        }
        if (this.biasVelocity != null) {
            this.biasVelocity = new SimpleMatrix(bias.numRows(), bias.numCols());
        }
    }
}
