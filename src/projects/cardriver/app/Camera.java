package projects.cardriver.app;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import projects.cardriver.entities.Car;

/**
 * This class contains positional information of where to render entities on the canvas.
 * It provides methods for car tracking and zooming.
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public class Camera
{
    private static final double SMOOTHING = 0.1;

    private Canvas canvas;
    private Car targetCar;

    private double xPos;
    private double yPos;
    private double scale = 1.0;

    /**
     * @param canvas the JavaFx canvas to draw on.
     */
    public Camera(Canvas canvas)
    {
        this.canvas = canvas;
        this.scale = 1.0;
    }

    /**
     * Updates the cameras position to the tracked car.
     */
    public void update()
    {
        if(targetCar != null)
        {
            xPos += (targetCar.getPosX() - xPos) * SMOOTHING * scale;
            yPos += (targetCar.getPosY() - yPos) * SMOOTHING * scale;
        }
    }

    /**
     * Translates and scales the scene according to the cameras position and zoom level.
     */
    public void startCapture()
    {
        double xCenter = -xPos + (canvas.getWidth() / 2 / scale) ;
        double yCenter = -yPos + (canvas.getHeight() / 2 / scale);
        canvas.getGraphicsContext2D().scale(scale, scale);
        canvas.getGraphicsContext2D().translate(xCenter, yCenter);
    }

    /**
     * Resets the scene translation and scaling.
     */
    public void endCapture()
    {
        double xCenter = xPos - (canvas.getWidth() / 2 / scale) ;
        double yCenter = yPos - (canvas.getHeight() / 2 / scale) ;
        canvas.getGraphicsContext2D().translate(xCenter, yCenter);
        canvas.getGraphicsContext2D().scale(1 / scale, 1 / scale);
    }

    /**
     * Changes the zoom level.
     * Positive values zooms in, negatives out.
     * @param dir the amount of change
     */
    public void zoom(double dir)
    {
        scale = Math.max(0.05, Math.min(2.0, scale + dir));
    }

    /**
     * Sets the car to track
     */
    public void track(Car car)
    {
        this.targetCar = car;
    }

    /**
     * @return the tracked car
     */
    public Car getTrackedCar()
    {
        return targetCar;
    }

    /**
     * @return the width of the viewport
     */
    public double getViewportWidth()
    {
        return canvas.getWidth();
    }

    /**
     * @return the height of the viewport
     */
    public double getViewportHeight()
    {
        return canvas.getHeight();
    }

    /**
     * @return the {@link GraphicsContext} object of the canvas
     */
    public GraphicsContext getGraphicsContext()
    {
        return canvas.getGraphicsContext2D();
    }
}
