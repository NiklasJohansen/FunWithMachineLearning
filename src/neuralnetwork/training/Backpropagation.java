package neuralnetwork.training;

import neuralnetwork.NeuralNetwork;
import neuralnetwork.Neuron;
import neuralnetwork.NeuronLayer;
import neuralnetwork.activationfunctions.ActivationFunction;

/**
 * This class provides a method of training the {@link NeuralNetwork} through supervised learning.
 * Samples of input data and ideal results is used to train the network iteratively.
 * The algorithm calculates the delta between the supplied ideal data and actual computed results,
 * and tunes the networks weights to minimize prediction error. It starts from the output neurons
 * and propagates backwards changing the weights through online training.
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public class Backpropagation extends NetworkTrainer
{
    private static final int MAX_ATTEMPTS = 10;

    private double[][][] prevWeightChange;
    private double[][][] gradients;
    private double[][] nodeDelta;

    private double learningRate;
    private double decayRate;
    private double momentum;
    private int resets;

    /**
     * The constructor accepts arrays containing normalized samples of inputs and ideal data, as
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
        this(inputData, idealData, learningRate, momentum, 0);
    }

    /**
     * The constructor accepts arrays containing normalized samples of inputs and ideal data, as
     * well as three training parameters.
     * @param inputData an array containing input data with one or more elements per sample.
     * @param idealData an array containing ideal data with one or more elements per sample.
     * @param learningRate a parameter specifying the rate of learning. Lower values makes the
     *                     learning go slower, but prevents oscillations yielding lower error rates.
     * @param momentum a parameter to help the algorithm overcome local minimas. A higher momentum may
     *                 increase the chance of reaching lower error rates, but can cause unwanted oscillation.
     * @param decayRate the amount of decay in the learning rate throughout the training process.
     */
    public Backpropagation(double[][] inputData, double[][]idealData, double learningRate, double momentum, double decayRate)
    {
        super(inputData, idealData);
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.decayRate = decayRate;
        this.batchSize = 1;
    }

    /**
     * Trains the network with the supplied dataset.
     * @param network the network to be trained
     * @param maxEpochs the number of training epoch. Higher numbers may yield lower error rates
     * @param acceptedError The accepted error rate in which the training will complete
     * @throws IllegalStateException if the network is not ready
     */
    @Override
    public void train(NeuralNetwork network, double acceptedError, int maxEpochs)
    {
        if(!network.isReady())
            throw new IllegalStateException("Training failed - network is not ready!");

        init(network);

        do
        {
            resets++;
            network.reset();

            for(epoch = 0; epoch < maxEpochs && meanSquaredError > acceptedError; epoch++)
            {
                meanSquaredError = executeEpoch(network);
                super.handleProgressCallback();
            }
        }
        while(meanSquaredError > acceptedError && resets < MAX_ATTEMPTS);
    }

    /**
     * Initiates all arrays.
     * @param network the network to be trained
     */
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
     * Iterates through the dataset changing the weights for each sample.
     * The error is calculated by summing up the squared delta for each sample and
     * dividing the result by the total number of samples (known as the Mean Squared Error).
     * @param network the network to be trained
     */
    private double executeEpoch(NeuralNetwork network)
    {
        NeuronLayer[] neuronLayers = network.getNeuronLayers();
        Neuron[] outputNeurons = network.getOutputLayer().getNeurons();
        ActivationFunction aFuncOut = network.getOutputLayer().getActivationFunction();

        double squaredErrorAccumulated = 0;
        int totalSampleCount = 0;
        double alpha = learningRate / (1.0 + decayRate * epoch);
        for(int sampleIdx = 0; sampleIdx < super.inputData.length; sampleIdx++)
        {
            boolean updateWeights = sampleIdx % batchSize == 0 || sampleIdx == super.inputData.length - 1;

            double[] trainingSample = super.inputData[sampleIdx];
            double[] idealResult = super.idealData[sampleIdx];
            double[] actualResult = network.compute(trainingSample);
            int nElements = Math.min(actualResult.length, idealResult.length);

            totalSampleCount += nElements;
            for(int i = 0; i < nElements; i++)
            {
                // Sums the mean squared error for every sample
                double deltaError = actualResult[i] - idealResult[i];
                squaredErrorAccumulated += deltaError * deltaError;

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
        return squaredErrorAccumulated / totalSampleCount;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected String getTrainingData()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("\nLearning rate: ").append(learningRate);
        sb.append("\nDecay rate: ").append(decayRate);
        sb.append("\nMomentum: ").append(momentum);
        sb.append("\nResets: ").append(resets);
        return sb.toString();
    }
}
