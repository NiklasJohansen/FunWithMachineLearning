package projects.cardriver.controllers;

import neuralnetwork.NeuralNetwork;

/**
 * This class extends the {@link CarController}.
 * It updates the gas and steering values by feeding the sensor inputs from
 * the car through a neural network.
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public class NNCarController extends CarController
{
    private NeuralNetwork neuralNetwork;

    /**
     * Creates a new neural network with tree layers consisting of six, five and two neurons.
     */
    public NNCarController()
    {
        this.neuralNetwork = new NeuralNetwork();
        this.neuralNetwork.addNeuronLayer(6); // Input layer
        this.neuralNetwork.addNeuronLayer(5); // Hidden layer
        this.neuralNetwork.addNeuronLayer(2); // Output layer
        this.neuralNetwork.build();
        this.neuralNetwork.reset();
    }

    /**
     * @param network the network to be used for computing gas and steering
     */
    public NNCarController(NeuralNetwork network)
    {
        this.neuralNetwork = network;
    }

    /**
     * @return the neural network of this controller
     */
    public NeuralNetwork getNeuralNetwork()
    {
        return neuralNetwork;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void update(double[] sensorInputs)
    {
        double[] outputs = neuralNetwork.compute(sensorInputs);
        super.steering = 2 * outputs[0] - 1;
        super.gas = 2 * outputs[1] - 1;
    }
}