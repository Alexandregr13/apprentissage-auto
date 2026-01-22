package fr.ensimag.deep.examples;

import java.io.FileWriter;
import java.io.IOException;
import org.ejml.simple.SimpleMatrix;

import com.opencsv.CSVWriter;

import fr.ensimag.deep.NeuralNetwork;
import fr.ensimag.deep.trainer.NetworkTrainer;
import fr.ensimag.deep.trainer.costFunction.QuadraticCostFunction;
import fr.ensimag.deep.trainer.dataShufflers.UniformShuffler;
import fr.ensimag.deep.utils.DataMatrix;

public class AndTrainerFromFile {
    public static void main(String args[]) throws IOException
    {
        // NeuralNetwork n = NeuralNetwork.load("Examples/and-trained.json");
        NeuralNetwork n = NeuralNetwork.fromDescription("and_network.json");

        SimpleMatrix inputs = new SimpleMatrix(new double[][] {
            {0, 0, 1, 1},
            {0, 1, 0, 1}
        } );

        SimpleMatrix expected_outputs = new SimpleMatrix(new double[][] {
            {0,0,0,1}
        });        

        DataMatrix data = new DataMatrix(inputs, expected_outputs);

        FileWriter outputfile = new FileWriter("error.csv"); 
  
        // create CSVWriter object filewriter object as parameter 
        CSVWriter writer = new CSVWriter(outputfile); 
  
        // adding header to csv 
        String[] header = { "Validation error"}; 
        writer.writeNext(header); 
        // closing writer connection 

        NetworkTrainer trainer = new NetworkTrainer(n, new QuadraticCostFunction(), new UniformShuffler());
        for (int i=0; i<35000; i++)
        {
            // next training epoch
            trainer.train(data);
            // training error on the data
            String[] val = {String.valueOf(trainer.validate(data))};
            // save the training error
            writer.writeNext(val);
        }
        writer.close(); 

        n.setBatchSize(4);
        n.propagate(inputs);
        System.out.println(inputs);
        System.out.println(n.getOutput());

        n.save("and_learned.json");        

        n.setBatchSize(4);
        n.propagate(inputs);
        System.out.println("Inputs:");
        System.out.println(inputs);
        System.out.println("Outputs");
        System.out.println(n.getOutput());
    }
}
