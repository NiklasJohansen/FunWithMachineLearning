package neuralnetwork.activationfunctions;

/**
 * This interface enables different activation functions.
 * Different activation functions is used to normalize the output of a neuron.
 * @author Niklas Johansen
 * @version 1.0
 */
public interface ActivationFunction
{
    /**
     * Passes the input through a given function and returns the result.
     * @param input the input value
     * @return the result of function
     */
    double compute(double input);

    /**
     * Passes the input through the derivative of the base function and returns the result.
     * @param input the input value
     * @return the result of function
     */
    double computeDerivative(double input);
}
