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
    private static final int PRINT_INTERVAL = 1000;

    private double[][] inputData;
    private double[][] idealData;
    private double learningRate;
    private double momentum;
    private double error;
    private double lowestError;
    private int randomWeightAttempts;
    private int iterations;
    private long timer;

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
     * Trains the network with the supplied dataset.
     * @param network the network to be trained
     * @param nIterations the number of training iterations. Higher numbers may yield lower error rates
     * @param acceptedError The accepted error rate in which the training will complete
     * @throws IllegalStateException if the network is not ready
     */
    public void trainNetwork(NeuralNetwork network, int nIterations, double acceptedError, boolean printProgress)
    {
        if(!network.isReady())
            throw new IllegalStateException("Training failed - network is not ready!");

        randomWeightAttempts = 0;

        do
        {
            lowestError = Double.MAX_VALUE;
            network.reset();

            for(iterations = 0; iterations < nIterations; iterations++)
            {
                error = trainSingleIteration(network);

                if(printProgress)
                    printTrainingProgress();

                if(error < acceptedError)
                    break;
            }
            randomWeightAttempts++;
        }
        while(error > acceptedError && randomWeightAttempts < MAX_ATTEMPTS);
    }

    /**
     * Iterates through the data set changing the weights for each sample.
     * The error is calculated by summing up the squared delta for each sample and
     * dividing the result by the total number of samples (known as the Mean Squared Error).
     * @param network the network to be trained
     */
    private double trainSingleIteration(NeuralNetwork network)
    {
        NeuronLayer[] neuronLayers = network.getNeuronLayers();

        double mse = 0;
        int samples = 0;
        for(int sampleIdx = 0; sampleIdx < inputData.length; sampleIdx++)
        {
            double[] trainingSample = inputData[sampleIdx];
            double[] idealResult = idealData[sampleIdx];
            double[] actualResult = network.compute(trainingSample);
            int nElements = Math.min(actualResult.length, idealResult.length);

            // Sums the mean squared error for every sample
            for(int i = 0; i < nElements; i++)
            {
                mse += Math.pow(idealResult[i] - actualResult[i], 2);
                samples++;
            }

            // Calculates the nodeDelta for the output neurons
            Neuron[] neurons = network.getOutputLayer().getNeurons();
            ActivationFunction activationFunction = network.getOutputLayer().getActivationFunction();
            for(int i = 0; i < nElements; i++)
            {
                double deltaError = actualResult[i] - idealResult[i];
                neurons[i].nodeDelta = -deltaError * activationFunction.computeDerivative(neurons[i].sum);
            }

            // Calculates the weight changes for each layer starting from the back
            for(int layerIdx = neuronLayers.length - 2; layerIdx >= 0; layerIdx--)
            {
                NeuronLayer thisLayer = neuronLayers[layerIdx];
                NeuronLayer nextLayer = neuronLayers[layerIdx + 1];

                Neuron[] nextLayerNeurons = nextLayer.getNeurons();

                for(Neuron neuron : thisLayer.getNeurons())
                {
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
        return mse / samples;
    }

    private void printTrainingProgress()
    {
        if(error < lowestError && System.currentTimeMillis() > timer + PRINT_INTERVAL)
        {
            System.out.println(iterations + ": " + error);
            lowestError = error;
            timer = System.currentTimeMillis();
        }
    }

    /**
     * @return a string containing information about the training session
     */
    public String getTrainingResultString()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("------------- Training Results -------------");
        sb.append("\nLearning rate: ").append(learningRate);
        sb.append("\nMomentum: ").append(momentum);
        sb.append("\nIterations: ").append(iterations);
        sb.append("\nRandom weight attempts: ").append(randomWeightAttempts);
        sb.append("\nMean squared error:  ").append(String.format("%.12f", error)).append("\n");
        return sb.toString();
    }

    /**
     * @return the mean squared error for the trained network
     */
    public double getError()
    {
        return error;
    }
}
