package projects.examples;

import neuralnetwork.NeuralNetwork;
import neuralnetwork.training.Backpropagation;

/**
 * The HelloWorld of neural networks - training it to predict the result of a XOR operator.
 * This class shows how the network can be trained with the {@link Backpropagation} class
 * on a small dataset.
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public class HelloWorld
{
    /**
     * Setts up a new network, trains it and computes four test results.
     * @param args input arguments
     */
    public static void main(String[] args)
    {
        NeuralNetwork network = new NeuralNetwork();
        network.addNeuronLayer(2); // Input layer
        network.addNeuronLayer(2); // Hidden layer
        network.addNeuronLayer(1); // Output layer
        network.build();

        double[][] inputData = {{0,0}, {1,0}, {0,1}, {1,1}};
        double[][] idealData = {{0},   {1},   {1},   {0}};

        Backpropagation trainer = new Backpropagation(inputData, idealData, 0.45, 1.0);
        trainer.trainNetwork(network, 100, 0.0001, false);
        System.out.println(trainer.getTrainingResultString() + "\n");

        System.out.println("0,0 = " + network.compute(0,0)[0]);
        System.out.println("1,0 = " + network.compute(1,0)[0]);
        System.out.println("0,1 = " + network.compute(0,1)[0]);
        System.out.println("1,1 = " + network.compute(1,1)[0]);
    }
}
