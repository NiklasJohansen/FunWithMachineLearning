package projects.examples;

import neuralnetwork.NeuralNetwork;
import neuralnetwork.datautils.AccuracyTester;
import neuralnetwork.datautils.ClassificationNormalizer;
import neuralnetwork.datautils.Dataset;
import neuralnetwork.training.Backpropagation;
import neuralnetwork.training.NetworkTrainer;

import java.io.IOException;
import java.util.Arrays;

/**
 * This example shows how the network can be trained to distinguish between
 * poisonous and edible mushrooms. More information on the data is available
 * on the linked website.
 *
 * The dataset is property of UCI Machine Learning Repository
 * Link: https://archive.ics.uci.edu/ml/datasets/mushroom
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public class MushroomClassification
{
    public static void main(String[] args) throws IOException
    {
        // Imports the dataset and adds it to the normalizer
        Dataset dataset = new Dataset(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data");
        ClassificationNormalizer normalizer = new ClassificationNormalizer();
        normalizer.addDataset(dataset.getTrainingSamples(), Dataset.ClassPosition.FIRST);
        System.out.println(normalizer);

        int nInputNeurons = normalizer.getNumberOfAttributes();
        int nOutputNeurons = normalizer.getNumberOfClasses();
        int nHiddenNeurons = nInputNeurons * 3 / 2;

        // Configures and builds the neural network
        NeuralNetwork network = new NeuralNetwork();
        network.addNeuronLayer(nInputNeurons);   // Input layer
        network.addNeuronLayer(nHiddenNeurons);  // Hidden layer
        network.addNeuronLayer(nOutputNeurons);  // Output layer
        network.build();

        // Creates and configures the trainer
        double[][][] trainingData = normalizer.getNormalizedTrainingData();
        NetworkTrainer trainer = new Backpropagation(trainingData[0], trainingData[1], 0.6, 0.7);
        trainer.setProgressCallbackAction(100, () ->
                System.out.println(trainer.getEpoch() + " " + trainer.getMeanSquaredError()));

        // Trains the network
        trainer.trainNetwork(network,0.00001, 1000);
        System.out.println(trainer.getTrainingResultString());

        // Tests the accuracy of the trained network on separate test samples
        AccuracyTester tester = new AccuracyTester(dataset.getTestSamples(), Dataset.ClassPosition.FIRST);
        tester.testClassification(network);
        System.out.println(tester.getTestResults());

        // Example classification
        String[] attributes = {"b","s","w","t","l","f","c","b","n","e","c","s","s","w","w","p","w","o","p","k","n","g"};
        double[] normalizedInput = normalizer.getNormalizedAttributes(attributes);
        double[] result = network.compute(normalizedInput);
        System.out.println("\nExample: " + Arrays.toString(attributes)
                + "  =>  " + normalizer.getClassMatchString(result));
    }
}
