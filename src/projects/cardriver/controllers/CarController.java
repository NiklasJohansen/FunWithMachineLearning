package projects.cardriver.controllers;

/**
 * This abstract class enables different controller implementations.
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public abstract class CarController
{
    protected double steering;
    protected double gas;

    /**
     * Called by the car to update the steering and gas values.
     * @param sensorInputs an array containing sensor values in the range from 0.0 to 1.0
     */
    public abstract void update(double[] sensorInputs);

    /**
     * @return the amount of steering in the range from -1.0 to 1.0
     */
    public double getSteering()
    {
        return steering;
    }

    /**
     * @return the amount of gas in the range from -1.0 to 1.0
     */
    public double getGas()
    {
        return gas;
    }
}
