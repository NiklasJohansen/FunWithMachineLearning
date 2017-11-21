package projects.examples;

import neuralnetwork.NeuralNetwork;
import neuralnetwork.datautils.AccuracyTester;
import neuralnetwork.datautils.ClassificationNormalizer;
import neuralnetwork.datautils.DataLoader;
import neuralnetwork.training.Backpropagation;

import java.io.IOException;
import java.util.Arrays;

/**
 * This example shows how the network is trained to precisely evaluate different cars.
 * A car can be evaluated to one of the following classes: unacc, acc, good, vgood
 * The evaluation is based upon six specific attributes:
 *  - buying price
 *  - maintenance price
 *  - number of doors
 *  - person capacity
 *  - luggage boot size
 *  - estimated safety
 *
 * The dataset is property of UCI Machine Learning Repository
 * Link: http://archive.ics.uci.edu/ml/datasets/Car+Evaluation
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public class CarEvaluation
{
    private NeuralNetwork network;
    private ClassificationNormalizer normalizer;

    public CarEvaluation() throws IOException
    {
        // Adds the dataset to the normalizer
        String[][] dataset = DataLoader.loadDatasetFromURL(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data");
        normalizer = new ClassificationNormalizer();
        normalizer.addDataset(dataset);
        System.out.println(normalizer);

        int nInputNeurons = normalizer.getNumberOfAttributes();
        int nOutputNeurons = normalizer.getNumberOfClasses();
        int nHiddenNeurons = nInputNeurons + nOutputNeurons;

        // Configures and builds the neural network
        network = new NeuralNetwork();
        network.addNeuronLayer(nInputNeurons);   // Input layer
        network.addNeuronLayer(nHiddenNeurons);  // Hidden layer
        network.addNeuronLayer(nOutputNeurons);  // Output layer
        network.build();

        // Trains the network
        double[][][] trainingData = normalizer.getNormalizedTrainingData();
        Backpropagation trainer = new Backpropagation(trainingData[0], trainingData[1], 0.3, 0.7);
        trainer.trainNetwork(network, 3000, 0.001, true);
        System.out.println("\n" + trainer.getTrainingResultString());

        // Tests the accuracy of the network on the same trainging data
        AccuracyTester tester = new AccuracyTester(dataset);
        System.out.println("Accuracy: " + tester.testClassification(network) + "%\n");

        // Evaluates different cars
        evaluate("low","low","5more","4","small","low");   // Ideal result: unacc
        evaluate("low","med","4","4","small","med");       // ideal result: acc
        evaluate("low","low","2","more","med","high");     // ideal reuslt: good
        evaluate("low","med","5more","more","big","high"); // ideal result: vgood
    }

    private void evaluate(String... values)
    {
        double[] normalizedInput = normalizer.getNormalizedAttributes(values);
        double[] result = network.compute(normalizedInput);
        System.out.println("Result: " + normalizer.getClassMatchString(result));
    }

    public static void main(String[] args) throws IOException
    {
        new CarEvaluation();
    }
}
