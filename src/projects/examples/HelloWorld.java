package projects.examples;

import neuralnetwork.Utils;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.training.Backpropagation;

import java.util.Arrays;
import java.util.Scanner;

/**
 * The HelloWorld of neural networks - training it to predict the result of a XOR operator.
 * This class shows how the network can be trained with the {@link Backpropagation} class
 * on an imported data set.
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public class HelloWorld
{
    /**
     * Setts up a new network, trains it with the imported data and computes a test result.
     * @param args input arguments
     */
    public static void main(String[] args)
    {
        NeuralNetwork network = new NeuralNetwork();
        network.addNeuronLayer(2); // Input layer
        network.addNeuronLayer(2); // Hidden layer 1
        network.addNeuronLayer(1); // Output layer
        network.build();

        double[][] input = Utils.importTrainingData("/trainingdata/xor_input.txt");
        double[][] ideal = Utils.importTrainingData("/trainingdata/xor_ideal.txt");

        Backpropagation trainer = new Backpropagation(input, ideal, 0.45, 1.0);
        trainer.trainNetwork(network, 1000, 0.001);

        double[] result = network.compute(1,0);
        System.out.println("1,0 = " + Arrays.toString(result));

        consoleInput(network);
    }

    /**
     * Reads and parses input from the IDE console, feeds it to the network
     * and prints out the result.
     * @param network the network to use
     */
    private static void consoleInput(NeuralNetwork network)
    {
        System.out.print("\nInput: ");
        Scanner scanner = new Scanner(System.in);
        while(scanner.hasNext())
        {
            String[] elements = scanner.next().split(",");
            double[] inputs = new double[elements.length];

            for(int i = 0; i < elements.length; i++)
                inputs[i] = Double.parseDouble(elements[i]);

            System.out.print(Arrays.toString(network.compute(inputs)) + "\nInput: ");
        }
    }
}
