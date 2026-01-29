package fr.ensimag.deep.utils;

import org.ejml.simple.SimpleMatrix;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.JsonParser;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

@NoArgsConstructor
public class DataNormalizer {

    @Getter @JsonProperty("mean")
    private double[] mean;

    @Getter @JsonProperty("std")
    private double[] std;

    @Getter @JsonProperty("numFeatures")
    private int numFeatures;

    private static final double EPSILON = 1e-8;

    public void fit(SimpleMatrix trainingInputs) {
        numFeatures = trainingInputs.getNumRows();
        int numSamples = trainingInputs.getNumCols();
        mean = new double[numFeatures];
        std = new double[numFeatures];

        for (int f = 0; f < numFeatures; f++) {
            double sum = 0.0;
            for (int s = 0; s < numSamples; s++) {
                sum += trainingInputs.get(f, s);
            }
            mean[f] = sum / numSamples;
        }

        for (int f = 0; f < numFeatures; f++) {
            double sumSq = 0.0;
            for (int s = 0; s < numSamples; s++) {
                double diff = trainingInputs.get(f, s) - mean[f];
                sumSq += diff * diff;
            }
            std[f] = Math.sqrt(sumSq / numSamples);
            if (std[f] < EPSILON) std[f] = 1.0;
        }
    }

    public SimpleMatrix transform(SimpleMatrix inputs) {
        int numSamples = inputs.getNumCols();
        SimpleMatrix normalized = new SimpleMatrix(numFeatures, numSamples);

        for (int f = 0; f < numFeatures; f++) {
            for (int s = 0; s < numSamples; s++) {
                normalized.set(f, s, (inputs.get(f, s) - mean[f]) / std[f]);
            }
        }
        return normalized;
    }

    public SimpleMatrix fitTransform(SimpleMatrix trainingInputs) {
        fit(trainingInputs);
        return transform(trainingInputs);
    }

    public void save(String filepath) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        Files.write(Paths.get(filepath), mapper.writerWithDefaultPrettyPrinter().writeValueAsString(this).getBytes());
    }

    public static DataNormalizer load(String filepath) throws IOException {
        byte[] jsonData = Files.readAllBytes(Paths.get(filepath));
        ObjectMapper mapper = new ObjectMapper().configure(JsonParser.Feature.ALLOW_COMMENTS, true);
        return mapper.readValue(jsonData, DataNormalizer.class);
    }
}
