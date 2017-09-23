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
 * @version 1.1
 * @see NeuronLayer
 * @see Neuron
 */
public class NeuralNetwork implements Serializable
{
    private static final int BIAS_NEURONS = 1;
    private static final double WEIGHT_INIT_RANGE = 2.0;

    private Random random;
    private NeuronLayer[] neuronLayers;
    private int nLayers;
    private boolean ready;

    public NeuralNetwork()
    {
        this.random = new Random();
        this.neuronLayers = new NeuronLayer[3];
    }

    /**
     * Adds a new layer to the network.
     * @param nNeurons the number of neurons in the layer
     * @param activationFunction the activation function to be used in calculation
     * @throws IllegalArgumentException if the number of neurons is zero or less
     */
    public void addNeuronLayer(int nNeurons, ActivationFunction activationFunction)
    {
        if(nNeurons > 0)
        {
            if(nLayers >= neuronLayers.length)
                neuronLayers = Arrays.copyOf(neuronLayers, (int)(neuronLayers.length * 1.5f));

            neuronLayers[nLayers++] = new NeuronLayer(nNeurons, activationFunction);
        }
        else throw new IllegalArgumentException("Each layer needs a minimum of one neuron!");
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
     * Builds the network structure as defined through the added layers.
     * This method has to be called to finalize the network before use.
     * @throws IllegalStateException if the network is not ready
     */
    public void build()
    {
        if(nLayers > 0)
        {
            neuronLayers = Arrays.copyOf(neuronLayers, nLayers);

            for(int i = 0; i < nLayers - 1; i++)
                neuronLayers[i].build(neuronLayers[i + 1].numberOfNormalNeurons(), BIAS_NEURONS);

            // No bias neurons or weights are needed for the output layer
            neuronLayers[nLayers - 1].build(0,0);

            ready = true;
        }
        else throw new IllegalStateException("Failed to build the network - no layers has been added!");
    }

    /**
     * Randomizes the networks weights and resets the training variables.
     * The initial weight range is constant and determined by WEIGHT_INIT_RANGE.
     * The network has to be built before it can be reset.
     * @throws IllegalStateException if the network is not ready
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
        else throw new IllegalStateException("Failed to reset - network is not built!");
    }

    /**
     * Feeds the supplied input data through the network and computes a result.
     * @param inputData an array containing data for each input neuron
     * @return an array containing the output of each output neuron
     * @throws IllegalStateException if the network is not ready
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

                int nNeurons = thisLayer.numberOfNormalNeurons();
                for(int neuronIdx = 0; neuronIdx < nNeurons; neuronIdx++)
                {
                    Neuron thisNeuron = thisLayer.getNeurons()[neuronIdx];

                    thisNeuron.sum = 0;
                    for(Neuron lastNeuron : lastLayer.getNeurons())
                        thisNeuron.sum += lastNeuron.output * lastNeuron.weights[neuronIdx];

                    thisNeuron.output = thisLayer.getActivationFunction().compute(thisNeuron.sum);
                }
            }
            return getOutputLayer().getOutputs();
        }
        else throw new IllegalStateException("Failed to compute - network is not built!");
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
     * @throws NoSuchElementException if no layers have been added
     */
    public NeuronLayer getInputLayer()
    {
        if(nLayers == 0)
            throw new NoSuchElementException("No input layer has been added!");

        return neuronLayers[0];
    }

    /**
     * @return the networks output layer
     * @throws NoSuchElementException if no layers have been added
     */
    public NeuronLayer getOutputLayer()
    {
        if(nLayers == 0)
            throw new NoSuchElementException("No output layer has been added!");

        return neuronLayers[nLayers - 1];
    }

    /**
     * The network is ready when layers have been added and the network is built.
     * @return a boolean indicating the state of the network
     */
    public boolean isReady()
    {
        return ready;
    }
}