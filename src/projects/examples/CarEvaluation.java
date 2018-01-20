package projects.examples;

import neuralnetwork.NeuralNetwork;
import neuralnetwork.datautils.AccuracyTester;
import neuralnetwork.datautils.ClassificationNormalizer;
import neuralnetwork.datautils.Dataset;
import neuralnetwork.training.NetworkTrainer;
import neuralnetwork.training.ResilientPropagation;

import java.io.IOException;
import java.util.Arrays;

/**
 * This example shows how the network is trained by the {@link ResilientPropagation} class,
 * to precisely evaluate different cars. A car can be evaluated to one of the following
 * classes: unacc, acc, good, vgood
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
        Dataset dataset = new Dataset("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data");
        normalizer = new ClassificationNormalizer();
        normalizer.addDataset(dataset.getSamples());
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

        // Creates and configures the trainer
        double[][][] trainingData = normalizer.getNormalizedTrainingData();
        NetworkTrainer trainer = new ResilientPropagation(trainingData[0], trainingData[1]);
        trainer.setProgressCallbackAction(100, () ->
                System.out.println(trainer.getEpoch() + " " + trainer.getMeanSquaredError()));

        // Trains the network
        trainer.trainNetwork(network,0.001, 10000);
        System.out.println(trainer.getTrainingResultString());

        /* Tests the accuracy of the network on the same training data.
           Larger datasets with more permutations should be tested on a separate testset. */
        AccuracyTester tester = new AccuracyTester(dataset.getSamples());
        tester.testClassification(network);
        System.out.println(tester.getTestResults());

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
        System.out.print("\n" + Arrays.toString(values) + "  =>  " + normalizer.getClassMatchString(result));
    }

    public static void main(String[] args) throws IOException
    {
        new CarEvaluation();
    }
}
