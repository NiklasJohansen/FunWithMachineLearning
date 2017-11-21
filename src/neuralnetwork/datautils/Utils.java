package neuralnetwork.datautils;

import neuralnetwork.NeuralNetwork;
import neuralnetwork.NeuronLayer;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * This class provides static utility methods for loading training data,
 * importing and exporting a {@link NeuralNetwork}, as well as printing
 * the a networks structure to the console.
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public class Utils
{
    /**
     * Imports a {@link NeuralNetwork} from a local file.
     * @param filename the file containing the network
     * @return a new NeuralNetwork instance
     */
    public static NeuralNetwork importNetwork(String filename)
    {
        try(ObjectInputStream objectInputStream = new ObjectInputStream(new FileInputStream(filename)))
        {
            return (NeuralNetwork) objectInputStream.readObject();
        }
        catch (IOException | ClassNotFoundException e)
        {
            e.printStackTrace();
        }

        return null;
    }

    /**
     * Exports a {@link NeuralNetwork} to a local file.
     * If the specified filename exists, an incrementing number is added to the end.
     * @param neuralNetwork the network to be exported
     * @param filename the name of the file. No file type is needed (default is .nn)
     */
    public static void exportNetwork(NeuralNetwork neuralNetwork, String filename)
    {
        String newFilename;
        for(int counter = 0; true; counter++)
            if(!(new File(newFilename = filename + "_" + counter + ".nn")).exists())
                break;

        try(ObjectOutputStream objectOutputStream = new ObjectOutputStream(new FileOutputStream(newFilename)))
        {
            objectOutputStream.writeObject(neuralNetwork);
            System.out.println("Network saved to: " + newFilename);
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    /**
     * Prints the structure of a {@link NeuralNetwork} with weights and outputs to the console.
     * @param network the network to be printed
     */
    public static void printNetworkStructure(NeuralNetwork network)
    {
        NeuronLayer[] neuronLayers = network.getNeuronLayers();

        for(int layerIdx = 0; layerIdx < neuronLayers.length; layerIdx++)
        {
            System.out.println("Layer_" + layerIdx + "");
            for (int n = 0; n < neuronLayers[layerIdx].getNeurons().length; n++)
            {
                System.out.println("  Neuron_" + n + " - output: " + neuronLayers[layerIdx].getNeurons()[n].output);
                for (int w = 0; w < neuronLayers[layerIdx].getNeurons()[n].weights.length; w++)
                {
                    System.out.println("    Weight_" + w + " = "  + neuronLayers[layerIdx].getNeurons()[n].weights[w]);
                }
            }
        }
    }
}
