package neuralnetwork.activationfunctions;

import java.io.Serializable;

/**
 * The Sigmoid activation function normalizes the output
 * of a neuron to a value between 0.0 and 1.0.
 *
 * @author Niklas Johansen
 * @version 1.0
 * @see ActivationFunction
 */
public class Sigmoid implements ActivationFunction, Serializable
{
    /**
     * {@inheritDoc}
     */
    @Override
    public double compute(double input)
    {
        return 1.0f / (1.0f + Math.exp(-input));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double computeDerivative(double input)
    {
        return compute(input) * (1.0f - compute(input));
    }
}
