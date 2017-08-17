package neuralnetwork;

import neuralnetwork.activationfunctions.ActivationFunction;

import java.io.Serializable;

/**
 * This class represents a neural layer and contains an array of
 * {@link Neuron neurons}, methods for accessing them and setting
 * their values. Each layer is supplied with its own activation function
 * which is used to normalize the output of the layers neurons.
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
    private ActivationFunction activationFunction;

    /**
     * Sets the activation function and creates the neuron array for this layer.
     * @param activationFunction the function to be used in the neurons output calculation
     * @param nNeurons the amount of normal neurons
     * @param nBiasNeurons the amount of bias neurons
     * @param nWeightsPerNeuron the number of weights going out of each neuron
     */
    public NeuronLayer(ActivationFunction activationFunction, int nNeurons, int nWeightsPerNeuron, int nBiasNeurons)
    {
        this.nBiasNeurons = nBiasNeurons;
        this.neurons = new Neuron[nNeurons + nBiasNeurons];
        this.activationFunction = activationFunction;

        // Normal neurons
        for(int i = 0; i < nNeurons; i++)
            this.neurons[i] = new Neuron(nWeightsPerNeuron);

        // Bias neurons
        for(int i = nNeurons; i < nNeurons + nBiasNeurons; i++)
            this.neurons[i] = new Neuron(nWeightsPerNeuron, 1.0);
    }

    /**
     * Sets the outputs of each neuron.
     * @param data an array containing data for each neuron.
     */
    public void setOutputs(double[] data)
    {
        if(data.length != neurons.length - nBiasNeurons)
            System.err.println("NB! The number of inputs (" + data.length
                    + ") don't match the number of neurons (" + (neurons.length - nBiasNeurons) + ") in this layer");

        for(int i = 0; i < Math.min(neurons.length, data.length); i++)
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
        return neurons.length - nBiasNeurons;
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