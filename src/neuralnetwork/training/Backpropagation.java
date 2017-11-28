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
    private static final int MAX_ATTEMPTS   = 10;
    private static final int PRINT_INTERVAL = 1000;

    private double[][] inputData;
    private double[][] idealData;

    private double[][][] prevWeightChange;
    private double[][][] gradients;
    private double[][] nodeDelta;

    private double learningRate;
    private double decayRate;
    private double momentum;
    private double error;
    private double lowestError;
    private int resets;
    private int batchSize;
    private int epoch;
    private long printTimer;
    private long trainingTime;

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
        this.batchSize = 1;
        this.decayRate = 0;
    }

    /**
     * Trains the network with the supplied dataset.
     * @param network the network to be trained
     * @param nIterations the number of training epoch. Higher numbers may yield lower error rates
     * @param acceptedError The accepted error rate in which the training will complete
     * @throws IllegalStateException if the network is not ready
     */
    public void trainNetwork(NeuralNetwork network, int nIterations, double acceptedError, boolean printProgress)
    {
        if(!network.isReady())
            throw new IllegalStateException("Training failed - network is not ready!");

        long startTime = System.currentTimeMillis();
        init(network);

        do
        {
            lowestError = Double.MAX_VALUE;
            resets++;
            network.reset();

            for(epoch = 0; epoch < nIterations; epoch++)
            {
                error = trainSingleIteration(network);

                if(printProgress)
                    printTrainingProgress();

                if(error < acceptedError)
                    break;
            }
        }
        while(error > acceptedError && resets < MAX_ATTEMPTS);
        trainingTime = System.currentTimeMillis() - startTime;
    }

    private void init(NeuralNetwork network)
    {
        int nLayers = network.getNeuronLayers().length;
        this.prevWeightChange = new double[nLayers][][];
        this.gradients = new double[nLayers][][];
        this.nodeDelta = new double[nLayers][];
        for(int i = 0; i < nLayers; i++)
        {
            Neuron[] neurons = network.getNeuronLayers()[i].getNeurons();
            this.gradients[i] = new double[neurons.length][neurons[0].weights.length];
            this.prevWeightChange[i] = new double[neurons.length][neurons[0].weights.length];
            this.nodeDelta[i] = new double[neurons.length];
        }
        this.resets = -1;
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
        Neuron[] outputNeurons = network.getOutputLayer().getNeurons();
        ActivationFunction aFuncOut = network.getOutputLayer().getActivationFunction();

        double mse = 0;
        int samples = 0;
        double alpha = learningRate / (1.0 + decayRate * epoch);
        for(int sampleIdx = 0; sampleIdx < inputData.length; sampleIdx++)
        {
            boolean updateWeights = sampleIdx % batchSize == 0 || sampleIdx == inputData.length - 1;

            double[] trainingSample = inputData[sampleIdx];
            double[] idealResult = idealData[sampleIdx];
            double[] actualResult = network.compute(trainingSample);
            int nElements = Math.min(actualResult.length, idealResult.length);

            for(int i = 0; i < nElements; i++)
            {
                // Sums the mean squared error for every sample
                double deltaError = actualResult[i] - idealResult[i];
                mse += deltaError * deltaError;
                samples++;

                // Calculates the nodeDelta for the output neurons
                nodeDelta[neuronLayers.length - 1][i] = -deltaError * aFuncOut.computeDerivative(outputNeurons[i].sum);
            }

            // Calculates the weight changes for each layer starting from the back
            for(int layerIdx = neuronLayers.length - 2; layerIdx >= 0; layerIdx--)
            {
                ActivationFunction aFunc = neuronLayers[layerIdx].getActivationFunction();
                Neuron[] thisLayerNeurons = neuronLayers[layerIdx].getNeurons();
                double[] nextLayerNodeDelta = nodeDelta[layerIdx + 1];
                double[] thisLayerNodeDelta = nodeDelta[layerIdx];
                double[][] thisLayerGradients = gradients[layerIdx];

                for(int neuronIdx = 0; neuronIdx < thisLayerNeurons.length; neuronIdx++)
                {
                    Neuron neuron = thisLayerNeurons[neuronIdx];
                    double[] neuronGradients = thisLayerGradients[neuronIdx];
                    double[] pwc = prevWeightChange[layerIdx][neuronIdx];

                    double weightSum = 0;
                    for(int i = 0; i < neuron.weights.length; i++)
                        weightSum += neuron.weights[i] * nextLayerNodeDelta[i];

                    thisLayerNodeDelta[neuronIdx] = weightSum * aFunc.computeDerivative(neuron.sum);

                    for(int i = 0; i < neuron.weights.length; i++)
                    {
                        neuronGradients[i] += nextLayerNodeDelta[i] * neuron.output;
                        if(updateWeights)
                        {
                            double deltaWeight = alpha * neuronGradients[i] + momentum * pwc[i];
                            pwc[i] = deltaWeight;
                            neuron.weights[i] += deltaWeight;
                            neuronGradients[i] = 0;
                        }
                    }
                }
            }
        }
        return mse / samples;
    }

    private void printTrainingProgress()
    {
        if(error < lowestError && System.currentTimeMillis() > printTimer + PRINT_INTERVAL)
        {
            System.out.println(epoch + ": " + error);
            lowestError = error;
            printTimer = System.currentTimeMillis();
        }
    }

    /**
     *
     * @param size
     */
    public void setBatchSize(int size)
    {
        this.batchSize = size;
    }

    /**
     * The decay rate
     * @param rate
     */
    public void setDecayRate(double rate)
    {
        this.decayRate = rate;
    }

    /**
     * @return a string containing information about the training session
     */
    public String getTrainingResultString()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("\n------------- Training Results -------------");
        sb.append("\nTraining samples: ").append(inputData.length);
        sb.append("\nMini-batch size: ").append(batchSize == 1 ? "1 (stochastic)" : batchSize);
        sb.append("\nLearning rate: ").append(learningRate);
        sb.append("\nDecay rate: ").append(decayRate);
        sb.append("\nMomentum: ").append(momentum);
        sb.append("\nEpochs: ").append(epoch);
        sb.append("\nResets: ").append(resets);
        sb.append("\nTraining time: " + trainingTime + " ms");
        sb.append("\nMean squared error:  ").append(String.format("%.12f", error));
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
