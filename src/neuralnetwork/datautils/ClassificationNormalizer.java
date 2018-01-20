package neuralnetwork.datautils;

import java.util.ArrayList;
import java.util.Arrays;

import static neuralnetwork.datautils.Dataset.ClassPosition;

/**
 * Enables importing and normalizing of arbitrary data from a comma-separated dataset.
 * Scans through the given data and collects all categorical/continuous attributes and classes.
 * Used to normalize the data into numerical values between -1.0 and 1.0.
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public class ClassificationNormalizer
{
    private double normalLow = -1.0;
    private double normalHigh = 1.0;

    private ArrayList<Attribute> attributes;
    private ArrayList<String> classes;
    private String[][] dataset;
    private int classIndex;

    public ClassificationNormalizer()
    {
        this.attributes = new ArrayList<>();
        this.classes = new ArrayList<>();
    }

    /**
     * Adds the dataset to the normalizer.
     * The last element in each sample is expected to be the class.
     * @param dataset an array containing data samples
     */
    public void addDataset(String[][] dataset)
    {
        addDataset(dataset, ClassPosition.LAST);
    }

    /**
     * Adds the dataset and specifies the class index.
     * @param dataset an array containing data samples
     * @param position the position of the class elements among the sample elements
     */
    public void addDataset(String[][] dataset, ClassPosition position)
    {
        if(dataset == null || dataset.length == 0)
            throw new IllegalArgumentException("Empty dataset!");

        this.dataset = dataset;
        this.classIndex = (position == ClassPosition.FIRST ? 0 : dataset[0].length - 1);
        findAttributesAndClasses();
    }

    /**
     * Scans through all samples and collects the different attribute
     * categories and continuous number ranges.
     */
    private void findAttributesAndClasses()
    {
        int nElements = dataset[0].length;
        boolean[] isNumeric = new boolean[nElements];
        ArrayList<String>[] dataTypeList = new ArrayList[nElements];
        for(int i = 0; i < nElements; i++)
        {
            dataTypeList[i] = new ArrayList<>();
            isNumeric[i] = true;
        }

        for(String[] sample : dataset)
        {
            for(int j = 0; j < sample.length; j++)
            {
                String element = sample[j];
                if(!dataTypeList[j].contains(element))
                {
                    dataTypeList[j].add(element);
                    if(isNumeric[j] && !element.matches("^[\\d\\-\\.]+$"))
                        isNumeric[j] = false;
                }
            }
        }

        for(int i = 0; i < nElements; i++)
        {
            if(i == classIndex)
                classes = dataTypeList[i];
            else if(isNumeric[i])
            {
                double low = Double.MAX_VALUE;
                double high = Double.MIN_VALUE;
                for(String category : dataTypeList[i])
                {
                    double number = Double.parseDouble(category);
                    low  = Math.min(number, low);
                    high = Math.max(number, high);
                }
                attributes.add(new Attribute(low, high));
            }
            else attributes.add(new Attribute(dataTypeList[i].toArray()));
        }
    }

    /**
     * Generates a normalized training set for the neural network.
     * Array at index 0 contains the input attributes.
     * Array at index 1 contains the ideal output results.
     * @return an array containing a set of input- and ideal-data.
     */
    public double[][][] getNormalizedTrainingData()
    {
        if(dataset == null)
            throw new IllegalStateException("No dataset added!");

        double[][] inputData = new double[dataset.length][attributes.size()];
        double[][] idealData = new double[dataset.length][classes.size()];

        for(int sampleIdx = 0; sampleIdx < dataset.length; sampleIdx++)
        {
            String[] elements = dataset[sampleIdx];
            for(int i = 0, attrIdx = 0; i < elements.length; i++)
                if(i != classIndex)
                    inputData[sampleIdx][attrIdx] = getNormalizedValue(attributes.get(attrIdx++), elements[i]);

            int index = classes.indexOf(elements[classIndex]);
            idealData[sampleIdx][index] = 1.0;
        }

        return new double[][][] {inputData, idealData};
    }

    /**
     * Normalizes a string value(categorical or continuous) to a number between the
     * defined low and high points (default between -1.0 to 1.0).
     * @param attribute the associated attribute object
     * @param dataValue the value to normalize
     * @return a normalized number
     */
    private double getNormalizedValue(Attribute attribute, String dataValue)
    {
        if(attribute.isCategorical())
        {
            Object[] values = attribute.categories;
            for(double i = 0; i < values.length; i++)
                if(values[(int)i].equals(dataValue))
                    return normalLow + (normalHigh - normalLow) * (i + 0.5) / values.length;

            throw new IllegalArgumentException(dataValue + " was not found among the defined categories!");
        }
        else // Continuous
        {
            double value = Double.parseDouble(dataValue) - attribute.minRange;
            double delta = attribute.maxRange - attribute.minRange;
            return normalLow + (normalHigh - normalLow) * (value / delta);
        }
    }

    /**
     * Creates a normalized array of the given input attributes.
     * @param inputAttributes string values matching the dataset attributes
     * @return an equally large array with numbers ranging from 0.0 to 1.0.
     */
    public double[] getNormalizedAttributes(String... inputAttributes)
    {
        int length = Math.min(inputAttributes.length, attributes.size());
        double[] normData = new double[length];

        for(int i = 0; i < length; i++)
            normData[i] = getNormalizedValue(attributes.get(i), inputAttributes[i]);

        return normData;
    }

    /**
     * Finds the largest value in the supplied array and returns the corresponding class.
     * @param data an array containing normalized data
     * @return the corresponding class string
     */
    public String getTopClass(double[] data)
    {
        int length = Math.min(data.length, classes.size());

        int highestIndex = 0;
        double highestValue = data[0];
        for(int i = 1; i < length; i++)
        {
            if(data[i] > highestValue)
            {
                highestValue = data[i];
                highestIndex = i;
            }
        }
        return classes.get(highestIndex);
    }

    /**
     * Builds a string with all the available classes and match percentages.
     * @param data a normalized array of numbers related to each class
     * @return a formated string
     */
    public String getClassMatchString(double[] data)
    {
        StringBuilder sb = new StringBuilder();
        int length = Math.min(data.length, classes.size());

        for(int i = 0; i < length; i++)
        {
            sb.append(classes.get(i)).append('(');
            sb.append((int)(data[i]*100));
            sb.append("%)").append(' ');
        }

        return sb.toString();
    }

    /**
     * @return a string containing the detected attributes and classes
     */
    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < attributes.size(); i++)
        {
            Attribute a = attributes.get(i);
            sb.append("\nAttr ").append(i).append(": ");
            if(a.isCategorical())
                sb.append(Arrays.toString(a.categories));
            else
                sb.append("continuous (").append(a.minRange).append(" - ").append(a.maxRange).append(')');
        }
        sb.append("\nClasses: ").append(Arrays.toString(classes.toArray())).append('\n');
        return sb.toString();
    }

    /**
     * Sets the normalization range.
     * @param low the lowest point of a normalized attribute
     * @param high the highest point a normalized attribute
     */
    public void setNormalRange(double low, double high)
    {
        this.normalLow = low;
        this.normalHigh = high;
    }

    /**
     * @return the number of attributes in the added dataset
     */
    public int getNumberOfAttributes()
    {
        return attributes.size();
    }

    /**
     * @return the number of classes in the dataset
     */
    public int getNumberOfClasses()
    {
        return classes.size();
    }

    /**
     * A private class containing data related to an input attribute.
     */
    private class Attribute
    {
        private double minRange, maxRange;
        private Object[] categories;

        private Attribute(Object[] v)
        {
            this.categories = v;
        }

        private Attribute(double minRange, double maxRange)
        {
            this.minRange = minRange;
            this.maxRange = maxRange;
        }

        public boolean isCategorical()
        {
            return categories != null;
        }
    }
}