package neuralnetwork;

import neuralnetwork.activationfunctions.ActivationFunction;
import neuralnetwork.activationfunctions.Sigmoid;
import neuralnetwork.training.*;
import java.io.Serializable;
import java.util.*;

/**
 * This class contains the data structure of a feed forward neural network. It provides
 * methods for building and accessing the structure, as well as computing results from
 * supplied input. The networks internal structure is made visible through getters to
 * allow for training with the {@link Backpropagation} and {@link GeneticAlgorithm} classes.
 *
 * @author Niklas Johansen
 * @version 1.0
 * @see NeuronLayer
 * @see Neuron
 */
public class NeuralNetwork implements Serializable
{
    private static final int BIAS_NEURONS = 1;
    private static final double WEIGHT_INIT_RANGE = 2.0;

    private Queue<LayerProperty> layerQueue;
    private NeuronLayer[] neuronLayers;
    private Random random;
    private boolean ready;

    public NeuralNetwork()
    {
        this.neuronLayers = new NeuronLayer[0];
        this.layerQueue = new LinkedList<>();
        this.random = new Random();
    }

    /**
     * Builds the network structure as defined through the added layers.
     * This method has to be called to finalize the network before use.
     */
    public void build()
    {
        if(layerQueue.size() > 0)
        {
            neuronLayers = new NeuronLayer[layerQueue.size()];

            int index = 0;
            while(layerQueue.size() > 1)
                neuronLayers[index++] = new NeuronLayer(
                        layerQueue.peek().aFunc,
                        layerQueue.poll().nNeurons,
                        layerQueue.peek().nNeurons,
                        BIAS_NEURONS);

            // No bias neurons or weights are needed for the output layer
            neuronLayers[index] = new NeuronLayer(layerQueue.peek().aFunc, layerQueue.poll().nNeurons, 0, 0);
            ready = true;
        }
        else System.err.println("No layers has been added to the network");
    }

    /**
     * Randomizes the networks weights and resets the training variables.
     * The initial weight range is constant and determined by WEIGHT_INIT_RANGE.
     * The network has to be built before it can be reset.
     */
    public void reset()
    {
        if(ready)
        {
            for(NeuronLayer layer : neuronLayers)
            {
                for(Neuron neuron : layer.getNeurons())
                {
                    for(int weightIdx = 0; weightIdx < neuron.weights.length; weightIdx++)
                    {
                        neuron.weights[weightIdx] = WEIGHT_INIT_RANGE * 2 * random.nextDouble() - WEIGHT_INIT_RANGE;
                        neuron.weightChange[weightIdx] = 0;
                        neuron.gradients[weightIdx] = 0;
                    }
                }
            }
        }
        else System.err.println("Build the network before resetting");
    }

    /**
     * Feeds the supplied input data through the network and computes a result.
     * @param inputData an array containing data for each input neuron
     * @return an array containing the output of each output neuron
     */
    public double[] compute(double... inputData)
    {
        if(ready)
        {
            getInputLayer().setOutputs(inputData);

            for(int layerIdx = 1; layerIdx < neuronLayers.length; layerIdx++)
            {
                NeuronLayer lastLayer = neuronLayers[layerIdx - 1];
                NeuronLayer thisLayer = neuronLayers[layerIdx];

                for(int neuronIdx = 0; neuronIdx < thisLayer.numberOfNormalNeurons(); neuronIdx++)
                {
                    Neuron neuron = thisLayer.getNeurons()[neuronIdx];

                    neuron.sum = 0;
                    for(Neuron lastNeuron : lastLayer.getNeurons())
                        neuron.sum += lastNeuron.output * lastNeuron.weights[neuronIdx];

                    neuron.output = thisLayer.getActivationFunction().compute(neuron.sum);
                }
            }
            return getOutputLayer().getOutputs();
        }

        System.err.println("Build the network before computing");
        return new double[0];
    }

    /**
     * The network is ready when layers have been added and the network is built.
     * @return a boolean indicating the state of the network
     */
    public boolean isReady()
    {
        return ready;
    }

    /**
     * Adds a new layer to the network.
     * The Sigmoid function is set as default.
     * @param nNeurons the number of neurons in the layer
     */
    public void addNeuronLayer(int nNeurons)
    {
        addNeuronLayer(nNeurons, new Sigmoid());
    }

    /**
     * Adds a new layer to the network.
     * @param nNeurons the number of neurons in the layer
     * @param activationFunction the activation function to be used in calculation
     */
    public void addNeuronLayer(int nNeurons, ActivationFunction activationFunction)
    {
        if(nNeurons > 0)
            layerQueue.add(new LayerProperty(nNeurons, activationFunction));
        else
            System.err.println("Each layer needs a minimum of one neuron");
    }

    /**
     * @return an array containing the networks layers
     */
    public NeuronLayer[] getNeuronLayers()
    {
        return neuronLayers;
    }

    /**
     * @return the networks input layer
     */
    public NeuronLayer getInputLayer()
    {
        return neuronLayers[0];
    }

    /**
     * @return the networks output layer
     */
    public NeuronLayer getOutputLayer()
    {
        return neuronLayers[neuronLayers.length - 1];
    }

    /**
     * This is a container class to hold the properties for each layer to
     * be added in the building process.
     */
    private class LayerProperty
    {
        private int nNeurons;
        private ActivationFunction aFunc;
        private LayerProperty(int nNeurons, ActivationFunction aFunc)
        {
            this.nNeurons = nNeurons;
            this.aFunc = aFunc;
        }
    }
}