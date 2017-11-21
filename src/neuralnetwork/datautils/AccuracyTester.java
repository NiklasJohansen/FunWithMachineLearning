package neuralnetwork.datautils;

import neuralnetwork.NeuralNetwork;

import java.util.Arrays;

/**
 * Tests the prediction accuracy of a trained neural network with a supplied testset.
 * Currently supports classification tasks.
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public class AccuracyTester
{
    private String[][] testset;
    private int classPosition;

    /**
     * @param testset a set of test samples
     */
    public AccuracyTester(String[][] testset)
    {
        this(testset, Integer.MAX_VALUE);
    }

    /**
     * @param testset a set of test samples
     * @param classPosition the position of the class in the testset
     */
    public AccuracyTester(String[][] testset, int classPosition)
    {
        if(testset == null || testset.length == 0)
            throw new IllegalArgumentException("Empty testset!");

        this.testset = testset;
        this.classPosition = Math.max(0, Math.min(classPosition, testset[0].length - 1));
    }

    /**
     * Computes result for every sample in the testset and counts the number of correct classifications
     * @param network the neural network to test
     * @return the accuracy in percentage
     */
    public float testClassification(NeuralNetwork network)
    {
        if(!network.isReady())
            throw new IllegalStateException("Network not ready!");

        ClassificationNormalizer normalizer = new ClassificationNormalizer();
        normalizer.addDataset(testset, classPosition);

        int hitCount = 0;

        for (String[] sample : testset)
        {
            String correctClass = sample[classPosition];

            if(classPosition == 0)
            {
                sample = Arrays.copyOfRange(sample,1, sample.length);
            }
            else if(classPosition < sample.length - 1)
            {
                String[] newSample = new String[sample.length - 1];
                for(int i = 0, j = 0; i < sample.length; i++)
                    if(i != classPosition)
                        newSample[j++] = sample[i];
                sample = newSample;
            }

            double[] result = network.compute(normalizer.getNormalizedAttributes(sample));
            String topClass = normalizer.getBestClassMatch(result);
            if (topClass.equals(correctClass))
                hitCount++;
        }

        return ((float)hitCount / testset.length) * 100;
    }
}
