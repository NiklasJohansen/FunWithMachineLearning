package neuralnetwork.training;

import neuralnetwork.NeuralNetwork;

/**
 * This abstract class provides methods for training the {@link NeuralNetwork}.
 * Normalized data is stored in arrays and used for training by the two currently available
 * training algorithms: {@link Backpropagation} and {@link ResilientPropagation}.
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public abstract class NetworkTrainer
{
    protected double[][] inputData;
    protected double[][] idealData;

    protected double meanSquaredError;
    protected int batchSize;
    protected int epoch;

    private Runnable progressCallback;
    private int callbackInterval = 1000;
    private long callbackTimer;
    private long trainingTime;

    public NetworkTrainer(double[][] inputData, double[][]idealData)
    {
        this.inputData = inputData;
        this.idealData = idealData;
        this.meanSquaredError = Double.MAX_VALUE;
    }

    protected abstract void train(NeuralNetwork network, double acceptedMeanSquaredError, int maxEpochs);

    /**
     * Trains the network on the given dataset.
     * @param network the neural network to be trained
     * @param acceptedMeanSquaredError the accepted error rate in which the training will complete.
     * @param maxEpochs the maximum number of training cycles. Higher numbers may yield lower error rates.
     */
    public void trainNetwork(NeuralNetwork network, double acceptedMeanSquaredError, int maxEpochs)
    {
        trainingTime = System.currentTimeMillis();
        train(network, acceptedMeanSquaredError, maxEpochs);
        trainingTime = System.currentTimeMillis() - trainingTime;
    }

    /**
     * This action will be called at a certain interval after each epoch is completed.
     * @param interval the time between each callback in milliseconds
     * @param callback the action to be run
     */
    public void setProgressCallbackAction(int interval, Runnable callback)
    {
        this.callbackInterval = interval;
        this.progressCallback = callback;
    }

    /**
     * Called by the subclasses and handles the timed progress callbacks.
     */
    protected void handleProgressCallback()
    {
        if(progressCallback != null && System.currentTimeMillis() > callbackTimer + callbackInterval)
        {
            progressCallback.run();
            callbackTimer = System.currentTimeMillis();
        }
    }

    /**
     * @param size the number of samples to accumulate weight change over, before the actual weights
     *             are changed. The default size is specified by the subclasses.
     */
    public void setBatchSize(int size)
    {
        this.batchSize = Math.max(1, size);
    }

    /**
     * @return the current training error
     */
    public double getMeanSquaredError()
    {
        return meanSquaredError;
    }

    /**
     * @return the current training epoch
     */
    public int getEpoch()
    {
        return epoch;
    }

    /**
     * @return a string containing training data special to the subclass
     */
    protected abstract String getTrainingData();

    /**
     * @return a string containing relevant training data
     */
    public String getTrainingResultString()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("\n------------- Training Results -------------");
        sb.append("\nTraining samples: ").append(inputData.length);
        sb.append("\nMini-batch size: ").append(batchSize == 1 ? "1 (stochastic)" : batchSize);
        sb.append(getTrainingData());
        sb.append("\nEpochs: ").append(epoch);
        sb.append("\nTraining time: ").append(trainingTime).append(" ms");
        sb.append("\nMean squared error:  ").append(String.format("%.12f", meanSquaredError));
        return sb.toString();
    }
}
