package neuralnetwork.training;

import neuralnetwork.NeuralNetwork;
import neuralnetwork.Neuron;
import neuralnetwork.NeuronLayer;
import neuralnetwork.activationfunctions.ActivationFunction;

/**
 * This class provides a method of training the {@link NeuralNetwork} through supervised learning.
 * Samples of input data and ideal results is used to train the network iteratively;
 * The algorithm calculates the delta between the supplied ideal data and actual computed results,
 * and tunes the networks weights to minimize prediction error. It starts from the output neurons
 * and propagates backwards changing the weights through online training.
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public class Backpropagation
{
    private static final int MAX_ATTEMPTS = 200;

    private double[][] inputData;
    private double[][] idealData;
    private double learningRate;
    private double momentum;
    private double error;

    /**
     * The constructor accepts arrays containing samples of inputs and ideal data, as
     * well as two training parameters.
     * @param inputData an array containing input data with one or more elements per sample.
     * @param idealData an array containing ideal data with one or more elements per sample.
     * @param learningRate a parameter specifying the rate of learning. Lower values makes the
     *                     learning go slower, but prevents oscillations yielding lower error rates.
     * @param momentum a parameter to help the algorithm overcome local minimas. A higher momentum may
     *                 increase the chance of reaching lower error rates, but can cause unwanted oscillation.
     */
    public Backpropagation(double[][] inputData, double[][]idealData, double learningRate, double momentum)
    {
        this.inputData = inputData;
        this.idealData = idealData;
        this.learningRate = learningRate;
        this.momentum = momentum;
    }

    /**
     * Trains the network with the supplied data set.
     * @param network the network to be trained
     * @param nIterations the number of training iterations. Higher numbers may yield lower error rates
     * @param acceptedError The accepted error rate in which the training will complete
     */
    public void trainNetwork(NeuralNetwork network, int nIterations, double acceptedError)
    {
        if(!network.isReady())
        {
            System.err.println("The network needs to be built before it can be trained.");
            return;
        }

        System.out.println("Training...\n");
        int randomWeightAttempts = 0;
        int iterations;

        do
        {
            network.reset();

            for(iterations = 0; iterations < nIterations; iterations++)
            {
                trainSingleIteration(network);
                if(error < acceptedError)
                    break;
            }

            randomWeightAttempts++;
        }
        while(error > acceptedError && randomWeightAttempts < MAX_ATTEMPTS);

        System.out.println("--------- Training Completed --------- ");
        System.out.println("Random weight attempts: " + randomWeightAttempts);
        System.out.println("Iterations: " + iterations);
        System.out.println("Success rate: " + (100.0 - error) + " %\n");
    }

    /**
     * Iterates through the data set changing the weights for each sample.
     * The error is calculated by summing up the squared delta for each sample and
     * dividing the result by the total number of samples (known as the Mean Squared Error).
     * @param network the network to be trained
     */
    private void trainSingleIteration(NeuralNetwork network)
    {
        NeuronLayer[] neuronLayers = network.getNeuronLayers();

        error = 0;
        int samples = 0;
        for(int sampleIdx = 0; sampleIdx < inputData.length; sampleIdx++)
        {
            double[] trainingSample = inputData[sampleIdx];
            double[] idealResult = idealData[sampleIdx];
            double[] actualResult = network.compute(trainingSample);

            // Sums the mean squared error for every sample
            for(int i = 0; i < Math.min(actualResult.length, idealResult.length); i++)
            {
                error += Math.pow(idealResult[i] - actualResult[i], 2);
                samples++;
            }

            // Calculates the nodeDelta for the output neurons
            Neuron[] neurons = network.getOutputLayer().getNeurons();
            ActivationFunction activationFunction = network.getOutputLayer().getActivationFunction();
            for(int i = 0; i < Math.min(actualResult.length, idealResult.length); i++)
            {
                double deltaError = actualResult[i] - idealResult[i];
                neurons[i].nodeDelta = -deltaError * activationFunction.computeDerivative(neurons[i].sum);
            }

            // Calculates the weight changes for each layer starting from the back
            for(int layerIdx = neuronLayers.length - 2; layerIdx >= 0; layerIdx--)
            {
                NeuronLayer thisLayer = neuronLayers[layerIdx];
                NeuronLayer nextLayer = neuronLayers[layerIdx + 1];

                for(Neuron neuron : thisLayer.getNeurons())
                {
                    Neuron[] nextLayerNeurons = nextLayer.getNeurons();

                    // Sums the weights going out of the neuron
                    double weightSum = 0;
                    for(int i = 0; i < neuron.weights.length; i++)
                        weightSum += neuron.weights[i] * nextLayerNeurons[i].nodeDelta;

                    // Calculates and setts the new node delta
                    neuron.nodeDelta = weightSum * thisLayer.getActivationFunction().computeDerivative(neuron.sum);

                    // Calculates the gradients and updates the weights through online training
                    for(int i = 0; i < neuron.weights.length; i++)
                    {
                        neuron.gradients[i] = nextLayerNeurons[i].nodeDelta * neuron.output;
                        double deltaWeight = learningRate * neuron.gradients[i] + momentum * neuron.weightChange[i];
                        neuron.weightChange[i] = deltaWeight;
                        neuron.weights[i] += deltaWeight;
                    }
                }
            }
        }
        error /= samples;
    }

    /**
     * @return the mean squared error for the trained network
     */
    public double getError()
    {
        return error;
    }
}
