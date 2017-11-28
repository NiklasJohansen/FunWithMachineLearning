package neuralnetwork;

import java.io.Serializable;

/**
 * This class contains the core data of a neuron.
 * Used as a plane data container.
 *
 * @author Niklas Johansen
 * @version 1.0
 * @see NeuronLayer
 * @see Neuron
 */
public class Neuron implements Serializable
{
    // Used for computing
    public double[] weights;
    public double output;
    public double sum;

    /**
     * @param nWeights the number of outgoing weights
     */
    public Neuron(int nWeights)
    {
        this(nWeights, 0.0);
    }

    /**
     * @param nWeights the number of outgoing weights
     * @param initialOutput the initial output of the neuron
     */
    public Neuron(int nWeights, double initialOutput)
    {
        this.output = initialOutput;
        this.weights = new double[nWeights];
    }
}

