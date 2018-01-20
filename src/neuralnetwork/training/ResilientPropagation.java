package neuralnetwork.training;

import neuralnetwork.NeuralNetwork;
import neuralnetwork.Neuron;
import neuralnetwork.NeuronLayer;
import neuralnetwork.activationfunctions.ActivationFunction;

import java.util.Arrays;
/**
 * This class provides a method of training the {@link NeuralNetwork} through supervised learning.
 * Samples of input data and ideal results is used to train the network iteratively.
 * The algorithm used is nearly identical to the standard back-propagation approach, but differs in how
 * each weight is updated.
 *
 * More about how this algorithm works can be found in this article:
 * https://visualstudiomagazine.com/Articles/2015/03/01/Resilient-Back-Propagation.aspx
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public class ResilientPropagation extends NetworkTrainer
{
    private static final double etaPlus = 1.2;
    private static final double etaMinus = 0.5;
    private static final double deltaMax = 50.0;
    private static final double deltaMin = 1.0E-6;

    private double[][] nodeDelta;
    private double[][][] prevDelta;
    private double[][][] gradientsAccumulated;
    private double[][][] prevGradientsAccumulated;

    private double squaredErrorAccumulated;
    private double totalSampleCount;

    /**
     * The constructor accepts arrays containing normalized samples of inputs and ideal data.
     * @param inputData an array containing input data with one or more elements per sample.
     * @param idealData an array containing ideal data with one or more elements per sample.
     */
    public ResilientPropagation(double[][] inputData, double[][]idealData)
    {
        super(inputData, idealData);
        super.batchSize = -1;
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

        for(epoch = 0; epoch < maxEpochs && meanSquaredError > acceptedError; epoch++)
        {
            executeEpoch(network);
            calculateError();
            super.handleProgressCallback();
        }
    }

    private void init(NeuralNetwork network)
    {
        network.reset();
        int nLayers = network.getNeuronLayers().length;

        super.batchSize = (batchSize == -1) ? inputData.length : batchSize;
        this.nodeDelta = new double[nLayers][];
        this.prevDelta = new double[nLayers - 1][][];
        this.gradientsAccumulated = new double[nLayers - 1][][];
        this.prevGradientsAccumulated = new double[nLayers - 1][][];

        for(int i = 0; i < nLayers; i++)
        {
            int nNeurons = network.getNeuronLayers()[i].getNeurons().length;
            int nWeights = network.getNeuronLayers()[i].getNeurons()[0].weights.length;

            this.nodeDelta[i] = new double[nNeurons];

            if(i < nLayers - 1)
            {
                this.prevDelta[i] = new double[nNeurons][nWeights];
                this.gradientsAccumulated[i] = new double[nNeurons][nWeights];
                this.prevGradientsAccumulated[i] = new double[nNeurons][nWeights];
                for(int j = 0; j < nNeurons; j++)
                {
                    Arrays.fill(this.prevDelta[i][j], 0.45);
                    Arrays.fill(this.prevGradientsAccumulated[i][j], 0.45);
                }
            }
        }
    }

    private void executeEpoch(NeuralNetwork network)
    {
        int iterations = Math.max(1, inputData.length / batchSize);

        for(int i = 0; i < iterations; i++)
        {
            int batchStart = i * batchSize;
            int batchEnd = Math.min(inputData.length, (i + 1) * batchSize);

            calculateGradients(network, batchStart, batchEnd);
            updateWeights(network);
        }
    }

    private void calculateGradients(NeuralNetwork network, int batchStart, int batchEnd)
    {
        ActivationFunction aFuncOut = network.getOutputLayer().getActivationFunction();
        Neuron[] outputNeurons = network.getOutputLayer().getNeurons();
        NeuronLayer[] neuronLayers = network.getNeuronLayers();
        int lastLayerIndex = neuronLayers.length - 1;

        for(int sampleIdx = batchStart; sampleIdx < batchEnd; sampleIdx++)
        {
            double[] trainingSample = inputData[sampleIdx];
            double[] idealResult = idealData[sampleIdx];
            double[] actualResult = network.compute(trainingSample);
            int nElements = Math.min(actualResult.length, idealResult.length);

            // Calculates the nodeDelta for the output neurons and accumulates the error
            totalSampleCount += nElements;
            for(int i = 0; i < nElements; i++)
            {
                double deltaError = actualResult[i] - idealResult[i];
                squaredErrorAccumulated += deltaError * deltaError;
                nodeDelta[lastLayerIndex][i] = -deltaError * aFuncOut.computeDerivative(outputNeurons[i].sum);
            }

            // Calculates the gradients for each layer starting from the back
            for(int layerIdx = lastLayerIndex - 1; layerIdx >= 0; layerIdx--)
            {
                ActivationFunction aFunc = neuronLayers[layerIdx].getActivationFunction();
                Neuron[] thisLayerNeurons = neuronLayers[layerIdx].getNeurons();
                double[] nextLayerNodeDelta = nodeDelta[layerIdx + 1];
                double[] thisLayerNodeDelta = nodeDelta[layerIdx];
                double[][] layerGradients = gradientsAccumulated[layerIdx];

                for(int neuronIdx = 0; neuronIdx < thisLayerNeurons.length; neuronIdx++)
                {
                    Neuron neuron = thisLayerNeurons[neuronIdx];

                    double weightSum = 0;
                    for(int i = 0; i < neuron.weights.length; i++)
                        weightSum += neuron.weights[i] * nextLayerNodeDelta[i];

                    thisLayerNodeDelta[neuronIdx] = weightSum * aFunc.computeDerivative(neuron.sum);

                    double[] neuronGradients = layerGradients[neuronIdx];
                    for(int i = 0; i < neuron.weights.length; i++)
                        neuronGradients[i] += nextLayerNodeDelta[i] * neuron.output;
                }
            }
        }
    }

    private void updateWeights(NeuralNetwork network)
    {
        NeuronLayer[] neuronLayers = network.getNeuronLayers();

        for(int layerIdx = 0; layerIdx < neuronLayers.length - 1; layerIdx++)
        {
            Neuron[] thisLayerNeurons = neuronLayers[layerIdx].getNeurons();
            double[][] layerGradients = gradientsAccumulated[layerIdx];
            double[][] layerPrevGradients = prevGradientsAccumulated[layerIdx];
            double[][] layerPrevDelta = prevDelta[layerIdx];

            for(int neuronIdx = 0; neuronIdx < thisLayerNeurons.length; neuronIdx++)
            {
                Neuron neuron = thisLayerNeurons[neuronIdx];
                double[] neuronGradients = layerGradients[neuronIdx];
                double[] neuronPrevGradients = layerPrevGradients[neuronIdx];
                double[] neuronPrevDelta = layerPrevDelta[neuronIdx];

                for(int i = 0; i < neuron.weights.length; i++)
                {
                    double gradient = neuronGradients[i];
                    double sign = gradient * neuronPrevGradients[i];
                    double delta;

                    if(sign > 0.0) // Same sign
                    {
                        delta = Math.min(neuronPrevDelta[i] * etaPlus, deltaMax);
                        double change = sign(gradient) * delta;
                        neuron.weights[i] += change;
                    }
                    else if (sign < 0.0) // Different sign
                    {
                        delta = Math.max(neuronPrevDelta[i] * etaMinus, deltaMin);
                        neuron.weights[i] -= neuronPrevDelta[i];
                        gradient = 0;
                    }
                    else // Last iteration was an overshoot
                    {
                        delta = neuronPrevDelta[i];
                        double change = sign(gradient) * delta;
                        neuron.weights[i] += change;
                    }

                    neuronPrevDelta[i] = delta;
                    neuronPrevGradients[i] = gradient;
                    neuronGradients[i] = 0;
                }
            }
        }
    }

    private double sign(double value)
    {
        return value < 0 ? -1 : (value > 0 ? 1 : 0);
    }

    private void calculateError()
    {
        super.meanSquaredError = squaredErrorAccumulated / totalSampleCount;
        totalSampleCount = 0;
        squaredErrorAccumulated = 0;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected String getTrainingData()
    {
        return "";
    }
}