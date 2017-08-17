package neuralnetwork;

import java.io.Serializable;

/**
 * This class contains the core data of a neuron.
 * All fields are public since the class is used as a basic data container.
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

    // Used for training
    public double[] weightChange;
    public double[] gradients;
    public double nodeDelta;

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
        this.gradients = new double[nWeights];
        this.weightChange = new double[nWeights];
    }
}

