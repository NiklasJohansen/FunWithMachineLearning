package neuralnetwork.activationfunctions;

/**
 * The hyperbolic tangent activation function normalizes the output
 * of a neuron to a value between -1.0 and 1.0.
 *
 * @author Niklas Johansen
 * @version 1.0
 * @see ActivationFunction
 */
public class HyperbolicTangent implements ActivationFunction
{
    /**
     * {@inheritDoc}
     */
    @Override
    public double compute(double input)
    {
        return Math.tanh(input);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double computeDerivative(double input)
    {
        double tanh = compute(input);
        return 1.0 - tanh * tanh;
    }
}
