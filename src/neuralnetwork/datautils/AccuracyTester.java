package neuralnetwork.datautils;

import neuralnetwork.NeuralNetwork;
import java.util.Arrays;
import static neuralnetwork.datautils.Dataset.ClassPosition;

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
    private ClassPosition classPosition;
    private int nCorrectClassifications;

    /**
     * @param testset a String array containing test samples
     */
    public AccuracyTester(String[][] testset)
    {
        this(testset, ClassPosition.LAST);
    }

    /**
     * @param testset a String array containing test samples
     * @param classPosition the position of the class in the set
     */
    public AccuracyTester(String[][] testset, ClassPosition classPosition)
    {
        if(testset == null || testset.length == 0)
            throw new IllegalArgumentException("Empty testset!");

        this.testset = testset;
        this.classPosition = classPosition;
    }

    /**
     * Computes results for every sample in the set and counts the number of correct classifications.
     * @param network the neural network to test
     * @return the accuracy in percentage
     * @throws IllegalStateException if the network is not ready
     */
    public float testClassification(NeuralNetwork network)
    {
        if(!network.isReady())
            throw new IllegalStateException("Network not ready!");

        ClassificationNormalizer normalizer = new ClassificationNormalizer();
        normalizer.addDataset(testset, classPosition);

        int classIndex = (classPosition == ClassPosition.FIRST ? 0 : testset[0].length - 1);
        nCorrectClassifications = 0;

        for (String[] sample : testset)
        {
            String correctClass = sample[classIndex];

            if(classIndex == 0)
            {
                sample = Arrays.copyOfRange(sample,1, sample.length);
            }
            else if(classIndex < sample.length - 1)
            {
                String[] newSample = new String[sample.length - 1];
                for(int i = 0, j = 0; i < sample.length; i++)
                    if(i != classIndex)
                        newSample[j++] = sample[i];
                sample = newSample;
            }

            double[] result = network.compute(normalizer.getNormalizedAttributes(sample));
            String topClass = normalizer.getTopClass(result);
            if (topClass.equals(correctClass))
                nCorrectClassifications++;
        }

        return ((float)nCorrectClassifications / testset.length) * 100;
    }

    /**
     * @return a string with relevant test results
     */
    public String getTestResults()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("\n------------- Test Results -------------");
        sb.append("\nTest samples: ").append(testset.length);
        sb.append("\nCorrect classifications: ").append(nCorrectClassifications);
        sb.append("\nAccuracy: ").append((float)nCorrectClassifications / testset.length * 100).append("%");
        return sb.toString();
    }

}
