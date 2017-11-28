package neuralnetwork.datautils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Arrays;


/**
 * Loads a dataset from a file or URL.
 * Requires individual data samples to be on separate lines and attributes to be comma-separated.
 * Provides methods for filtering out irrelevant data, as well as splitting the dataset into
 * sets for training and testing.
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public class Dataset
{
    public enum ClassPosition { FIRST, LAST }

    private final static String DELIMITER = ",";
    private final static int TRAINING_SET_PERCENTAGE = 80;

    private String[][] dataset;
    private boolean[] filter;
    private int nSampleElements;

    /**
     * Loads every data sample from a file into structured arrays.
     * @param path a local path or url.
     * @throws IllegalStateException if the loaded dataset is empty
     */
    public Dataset(String path)
    {
        try
        {
            URL url = new URL(path);
            this.dataset = load(new InputStreamReader(url.openConnection().getInputStream()));
        }
        catch (MalformedURLException e)
        {
            try
            {
                this.dataset = load(new InputStreamReader(Dataset.class.getResourceAsStream(path)));
            }
            catch (IOException e1)
            {
                e1.printStackTrace();
            }
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }

        if(dataset == null || dataset.length == 0)
            throw new IllegalStateException("Dataset is empty!");

        this.nSampleElements = dataset[0].length;
        this.filter = new boolean[nSampleElements];
        for(int i = 0; i < nSampleElements; i++)
            filter[i] = true;
    }

    private String[][] load(Reader reader) throws IOException
    {
        BufferedReader bReader = new BufferedReader(reader);
        String[][] dataSamples = new String[100][];
        int samples = 0;

        for(String line; (line = bReader.readLine()) != null;)
        {
            if(samples >= dataSamples.length)
                dataSamples = Arrays.copyOf(dataSamples, dataSamples.length * 2);

            String[] elements = line.split(DELIMITER);
            if(elements.length > 1)
            {
                for(int i = 0; i < elements.length; i++)
                    elements[i] = elements[i].trim();
                dataSamples[samples++] = elements;
            }
        }

        bReader.close();
        return Arrays.copyOf(dataSamples, samples - 1);
    }

    /**
     * Enables filtering of irrelevant elements in the loaded dataset.
     * The length of the filter has to match the number of sample elements.
     * @param filter an array of booleans
     * @throws IllegalArgumentException if the filter contains too few/many elements
     */
    public void setElementFilter(boolean... filter)
    {
        if(filter.length != nSampleElements)
            throw new IllegalArgumentException("Filter length(" + filter.length +
                    ") does not match the number of elements(" + nSampleElements + ")");

        this.filter = filter;
    }

    /**
     * @return a boolean array representing the active/included sample elements
     */
    public boolean[] getElementFilter()
    {
        return filter;
    }

    /**
     * @return a String array containing all loaded data samples
     */
    public String[][] getSamples()
    {
        return getSubset(0, dataset.length);
    }

    /**
     * Returns the first 80% of the samples in the loaded dataset.
     * @return a String array of samples
     */
    public String[][] getTrainingSamples()
    {
        int cutPoint = (int)(dataset.length * (TRAINING_SET_PERCENTAGE / 100.0));
        return getSubset(0, cutPoint);
    }

    /**
     * Returns the last 20% of the samples in the loaded dataset.
     * @return a String array of samples
     */
    public String[][] getTestSamples()
    {
        int cutPoint = (int)(dataset.length * (TRAINING_SET_PERCENTAGE / 100.0));
        return getSubset(cutPoint, dataset.length);
    }

    /**
     * Generates a subset of samples from the loaded dataset.
     * @throws IllegalArgumentException if the range is illegal
     */
    private String[][] getSubset(int from, int to)
    {
        int nSamples = to - from;
        if(from < 0 || to > dataset.length || nSamples < 1)
            throw new IllegalArgumentException("Illegal range!");

        int nActiveElements = 0;
        for(int i = 0; i < filter.length; i++)
            if(filter[i])
                nActiveElements++;

        if(nActiveElements == nSampleElements)
        {
            if(nSamples == dataset.length)
                return dataset;
            else
                return Arrays.copyOfRange(dataset, from, to);
        }
        else
        {
            String[][] subset = new String[nSamples][nActiveElements];
            for(int i = 0; i < nSamples; i++)
                for(int a = 0, j = 0; j < nSampleElements; j++)
                    if(filter[j])
                        subset[i][a++] = dataset[from + i][j];

            return subset;
        }
    }
}
