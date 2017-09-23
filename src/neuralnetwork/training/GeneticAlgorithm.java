package neuralnetwork.training;

import neuralnetwork.Neuron;
import neuralnetwork.NeuronLayer;
import neuralnetwork.NeuralNetwork;

/**
 * This class provides a method of combining the genes (weights) of two
 * {@link NeuralNetwork neural networks} to produce offspring with a new genetic composition.
 * The class uses a two-point crossover algorithm combined with a probability of mutation by
 * randomly swapping two genes.
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public class GeneticAlgorithm
{
    private static final float CUT_LENGTH_PERCENTAGE = 30;

    /**
     * Creates a new offspring network by combining the DNA from a mother and a father.
     * Uses a two-point crossover method with the cut length determined by the CUT_LENGTH_PERCENTAGE constant.
     * @param motherNetwork the mother network
     * @param fatherNetwork the father network
     * @param mutationProbability the chance of mutation in percentage
     * @return a new offspring network
     * @throws IllegalArgumentException if the networks has different internal structure
     */
    public NeuralNetwork breed(NeuralNetwork motherNetwork, NeuralNetwork fatherNetwork, double mutationProbability)
    {
        double[] motherDNA = getDNA(motherNetwork);
        double[] fatherDNA = getDNA(fatherNetwork);

        if(motherDNA.length != fatherDNA.length)
            throw new IllegalArgumentException("Breeding failed - networks have different internal structure!");

        int DNALength = motherDNA.length;
        int cutLength = (int)(DNALength * (CUT_LENGTH_PERCENTAGE / 100));
        int cutPoint1 = (int)(Math.random() * (DNALength - cutLength));
        int cutPoint2 = cutPoint1 + cutLength;

        double[] offspringDNA = new double[DNALength];

        for(int i = 0; i < DNALength; i++)
        {
            boolean betweenCutPoints = i > cutPoint1 && i < cutPoint2;
            offspringDNA[i] = betweenCutPoints ? motherDNA[i] : fatherDNA[i];
        }

        if(Math.random() * 100 < mutationProbability)
            swapMutateDNA(offspringDNA);

        return createOffspring(motherNetwork, offspringDNA);
    }

    /**
     * Creates a DNA string from the network.
     * @param network the network to create the string from
     * @return a one dimensional array containing the supplied networks weights
     */
    private double[] getDNA(NeuralNetwork network)
    {
        NeuronLayer[] layers = network.getNeuronLayers();

        int stringLength = 0;
        for(int i = 0; i < layers.length - 1; i++)
            stringLength += layers[i].getNeurons().length * layers[i + 1].getNeurons().length;

        double[] DNA = new double[stringLength];

        int index = 0;
        for(NeuronLayer layer : layers)
            for(Neuron neuron : layer.getNeurons())
                for(Double weight : neuron.weights)
                    DNA[index++] = weight;

        return DNA;
    }

    /**
     * Mutates the DNA by swapping two genes.
     * @param DNA the DNA string to be mutated
     */
    private void swapMutateDNA(double[] DNA)
    {
        int swapPoint1 = (int)(Math.random() * (DNA.length - 1));
        int swapPoint2 = (int)(Math.random() * (DNA.length - 1));

        double temp = DNA[swapPoint1];
        DNA[swapPoint1] = DNA[swapPoint2];
        DNA[swapPoint2] = temp;
    }

    /**
     * Creates an offspring network from the DNA with the same internal structure as the mother network
     * @param motherNetwork the network to copy the internal structure from
     * @param DNA the DNA string containing the weights
     * @return an offspring network
     */
    private NeuralNetwork createOffspring(NeuralNetwork motherNetwork, double[] DNA)
    {
        NeuralNetwork offspring = new NeuralNetwork();
        for(NeuronLayer layer : motherNetwork.getNeuronLayers())
            offspring.addNeuronLayer(layer.numberOfNormalNeurons());

        offspring.build();

        int index = 0;
        for(NeuronLayer layer : offspring.getNeuronLayers())
            for(Neuron neuron : layer.getNeurons())
                for(int i = 0; i < neuron.weights.length; i++)
                    neuron.weights[i] = DNA[index++];

        return offspring;
    }
}

