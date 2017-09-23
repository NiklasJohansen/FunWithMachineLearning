package neuralnetwork;

import neuralnetwork.activationfunctions.ActivationFunction;

import java.io.Serializable;

/**
 * This class represents a neural layer and contains an array of
 * {@link Neuron neurons}, methods for accessing them and setting
 * their values. Each layer is supplied with its own activation function
 * which is used to normalize the output of the neurons.
 *
 * @author Niklas Johansen
 * @version 1.0
 * @see NeuralNetwork
 * @see Neuron
 */
public class NeuronLayer implements Serializable
{
    private Neuron[] neurons;
    private int nBiasNeurons;
    private int nNormalNeurons;
    private ActivationFunction activationFunction;

    /**
     * Sets the activation function and neuron count.
     * @param activationFunction the function to be used in computing
     * @param nNeurons the amount of normal neurons
     */
    public NeuronLayer(int nNeurons, ActivationFunction activationFunction)
    {
        this.nNormalNeurons = nNeurons;
        this.activationFunction = activationFunction;
    }

    /**
     * Builds the layer by initializing the neuron array.
     * @param nWeightsPerNeuron the number of weights going out of each neuron
     * @param nBiasNeurons the number of bias neurons
     */
    public void build(int nWeightsPerNeuron, int nBiasNeurons)
    {
        this.nBiasNeurons = nBiasNeurons;
        this.neurons = new Neuron[nNormalNeurons + nBiasNeurons];

        // Normal neurons
        for(int i = 0; i < nNormalNeurons; i++)
            this.neurons[i] = new Neuron(nWeightsPerNeuron);

        // Bias neurons
        for(int i = nNormalNeurons; i < nNormalNeurons + nBiasNeurons; i++)
            this.neurons[i] = new Neuron(nWeightsPerNeuron, 1.0);
    }

    /**
     * Sets the outputs of each neuron.
     * @param data an array containing data for each neuron
     * @throws IllegalArgumentException if the number of data elements don't match the number of neurons
     */
    public void setOutputs(double[] data)
    {
        if(data.length != nNormalNeurons)
            throw new IllegalArgumentException("data(" + data.length + ") != neurons(" + nNormalNeurons + ")");

        for(int i = 0; i < nNormalNeurons; i++)
            neurons[i].output = data[i];
    }

    /**
     * @return an array containing the output value from each neuron
     */
    public double[] getOutputs()
    {
        double[] outputs = new double[neurons.length];
        for(int i = 0; i < outputs.length; i++)
            outputs[i] = neurons[i].output;
        return outputs;
    }

    /**
     * @return an array containing this layers neurons
     */
    public Neuron[] getNeurons()
    {
        return neurons;
    }

    /**
     * @return the amount of normal (not bias) neurons in this layer
     */
    public int numberOfNormalNeurons()
    {
        return nNormalNeurons;
    }

    /**
     * @return the number of bias neurons for this layer
     */
    public int numberOfBiasNeurons()
    {
        return nBiasNeurons;
    }

    /**
     * @return the activation function for this layer
     */
    public ActivationFunction getActivationFunction()
    {
        return activationFunction;
    }
}