package neuralnetwork.datautils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
import java.util.Arrays;

/**
 * Provides methods for loading a dataset from file or URL.
 * Requires individual data samples to be on separate lines and
 * attributes to be comma-separated.
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public class DataLoader
{
    private final static String DELIMITER = ",";

    /**
     * Reads a comma-separated file into a String array.
     * @param filename the file containing the dataset
     * @return an array containing one individual data sample per row
     */
    public static String[][] loadDatasetFromFile(String filename)
    {
        try
        {
            return load(new InputStreamReader(DataLoader.class.getResourceAsStream(filename)));
        }
        catch (IOException e)
        {
            e.printStackTrace();
            return new String[0][0];
        }
    }

    /**
     * Reads a comma-separated file into a String array.
     * @param filename the file containing the dataset
     * @return an array containing one individual data sample per row
     */
    public static String[][] loadDatasetFromURL(String filename)
    {
        try
        {
            URL destination = new URL(filename);
            return load(new InputStreamReader(destination.openConnection().getInputStream()));
        }
        catch (IOException e)
        {
            e.printStackTrace();
            return new String[0][0];
        }
    }

    private static String[][] load(Reader reader) throws IOException
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
}
