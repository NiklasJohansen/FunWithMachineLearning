package projects.cardriver.app;

import neuralnetwork.Neuron;
import neuralnetwork.NeuronLayer;
import neuralnetwork.NeuralNetwork;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

/**
 * This class provides a method for visually graphing out a {@link NeuralNetwork}.
 * Neurons are drawn as round ellipses and neuron connection as the lines between them.
 * The line thickness represents the connection strength (the weight) between two neurons.
 * Neurons are colored from black to white, representing the output strength.
 *
 * @author Niklas Johansen
 * @version 1.0
 */

public class NetworkGraph
{
    private static final int NEURON_SIZE    = 20;
    private static final int VERTICAL_GAP   = 10;
    private static final int HORIZONTAL_GAP = 80;
    private static final int BOX_BORDER     = 15;

    private int height;
    private int width;

    /**
     * Renders the network graph to the supplied canvas.
     * @param network the network to graph
     * @param canvas the canvas to draw the graph on
     * @param xPos the horizontal position
     * @param yPos the vertical position
     */
    public void render(NeuralNetwork network, Canvas canvas, int xPos, int yPos)
    {
        GraphicsContext gc =  canvas.getGraphicsContext2D();
        NeuronLayer[]  layers = network.getNeuronLayers();

        width = layers.length;
        height = 0;
        double halfNeuronSize = NEURON_SIZE / 2;

        for(NeuronLayer layer : network.getNeuronLayers())
            height = Math.max(height, layer.getNeurons().length);

        height = height * (NEURON_SIZE + VERTICAL_GAP);
        width  = (width  * NEURON_SIZE) + ((width - 1)  * HORIZONTAL_GAP);
        yPos += height / 2;

        // ------------------------------- Background box -------------------------------

        gc.setFill(Color.gray(0.6, 0.4));
        gc.setStroke(Color.BLACK);
        gc.setLineWidth(0.5);
        gc.fillRoundRect(
                xPos   - BOX_BORDER - halfNeuronSize,
                yPos   - BOX_BORDER - halfNeuronSize - (height / 2),
                width  + BOX_BORDER * 2,
                height + BOX_BORDER * 2 - halfNeuronSize,
                20, 20);
        gc.strokeRoundRect(
                xPos   - BOX_BORDER - halfNeuronSize,
                yPos   - BOX_BORDER - halfNeuronSize - (height / 2),
                width  + BOX_BORDER * 2,
                height + BOX_BORDER * 2 - halfNeuronSize,
                20, 20);

        // ------------------------------- Neurons and weight-lines -------------------------------

        for(int layerIdx = 0; layerIdx < layers.length; layerIdx++)
        {
            int nNeurons = layers[layerIdx].getNeurons().length;
            int yNeuronCenter = (nNeurons * (NEURON_SIZE + VERTICAL_GAP)) / 2;

            for(int neuronIdx = 0; neuronIdx < nNeurons; neuronIdx++)
            {
                Neuron neuron = layers[layerIdx].getNeurons()[neuronIdx];
                float xNeuron = xPos + layerIdx * (NEURON_SIZE + HORIZONTAL_GAP);
                float yNeuron = yPos + neuronIdx * (NEURON_SIZE + VERTICAL_GAP) - yNeuronCenter;

                if(layerIdx + 1 < layers.length)
                {
                    int nNextLayerNeurons = layers[layerIdx + 1].numberOfNormalNeurons();
                    int yNextNeuronCenter = ((nNextLayerNeurons + layers[layerIdx + 1].numberOfBiasNeurons())
                            * (NEURON_SIZE + VERTICAL_GAP)) / 2;

                    for(int nextNeuronIdx = 0; nextNeuronIdx < nNextLayerNeurons; nextNeuronIdx++)
                    {
                        float xNextNeuron = xPos + (layerIdx + 1) * (NEURON_SIZE + HORIZONTAL_GAP);
                        float yNextNeuron = yPos + nextNeuronIdx  * (NEURON_SIZE + VERTICAL_GAP) - yNextNeuronCenter;

                        gc.setStroke(neuron.weights[nextNeuronIdx] < 0 ? Color.RED : Color.GREEN);
                        gc.setLineWidth(Math.abs(neuron.output * neuron.weights[nextNeuronIdx]));
                        gc.strokeLine(xNeuron, yNeuron, xNextNeuron, yNextNeuron);
                    }
                }

                // Neurons
                float r = NEURON_SIZE / 2;
                double c = Math.abs(neuron.output);
                gc.setFill(Color.color(c,c,c));
                gc.setLineWidth(1.0);
                gc.setStroke(Color.BLACK);
                gc.fillRoundRect(xNeuron - r, yNeuron - r, NEURON_SIZE, NEURON_SIZE, NEURON_SIZE, NEURON_SIZE);
                gc.strokeRoundRect(xNeuron - r, yNeuron - r, NEURON_SIZE, NEURON_SIZE, NEURON_SIZE, NEURON_SIZE);
            }
        }
    }

    /**
     * @return the width of the graph
     */
    public int getWidth()
    {
        return width + 2 * BOX_BORDER;
    }

    /**
     * @return the height of the graph
     */
    public int getHeight()
    {
        return height + 2 * BOX_BORDER;
    }
}
