package projects.cardriver.entities;

import projects.cardriver.app.Camera;
import projects.cardriver.app.CarBreeder;
import projects.cardriver.controllers.CarController;
import projects.cardriver.controllers.NNCarController;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.effect.DropShadow;
import javafx.scene.effect.Effect;
import javafx.scene.paint.Color;
import javafx.scene.transform.Affine;
import javafx.scene.transform.Rotate;
import projects.cardriver.controllers.UserCarController;


/**
 * This class contains all values and parameters for a car, methods for driving it, tracking its progress
 * along a track, calculating its sensor values and rendering it to the screen.
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public class Car implements Comparable<Car>
{
    private static final double SENSOR_ANGLE_RANGE         =  90;
    private static final double SENSOR_STICK_LENGTH        =  300;
    private static final double UPDATES_WITHOUT_PROGRESS   =  120;
    private static final double ANGULAR_TRACTION_FACTOR    = 0.75;
    private static final double TRACTION_FACTOR            = 0.75;
    private static final double AIR_RESISTANCE             = 0.95;
    private static final double MAX_STEERING               =  0.5;
    private static final double MAX_GAS                    =  0.5;

    private final CarController controller;

    private double xPos;
    private double yPos;
    private double xVel;
    private double yVel;
    private double rot;
    private double rotVel;
    private double maxSpeed;
    private double length;
    private double width;

    private Effect dropShadow;
    private Color carColor;

    private double[] sensorValues;
    private double[][] sensorPoints;
    private double updatesSinceLastProgress;
    private double acquiredFitness;
    private double currentFitness;
    private int wayPointsPassed;
    private boolean finished;


    /**
     * Sets car parameters, adds sensors and determines max speed.
     * @param xPos the horizontal starting position
     * @param yPos the vertical starting position
     * @param controller the controller to be used
     */
    public Car(double xPos, double yPos, CarController controller)
    {
        this.carColor = (controller instanceof UserCarController) ? Color.BLUE : Color.color(1.0, 0.15, 0.15);
        this.dropShadow = new DropShadow(10, Color.gray(0.1, 0.5));
        this.controller = controller;
        this.xPos = xPos;
        this.yPos = yPos;
        this.length = 50;
        this.width = 25;
        this.rot = 180;

        // Number of sensors is determined by the controller. Default is seven.
        int sensors = 5;
        if(controller instanceof NNCarController)
            sensors = ((NNCarController) controller).getNeuralNetwork().getInputLayer().numberOfNormalNeurons();
        this.sensorValues = new double[sensors];
        this.sensorPoints = new double[sensors][4];

        // Calculates max speed by numerical approximation
        for(int i = 0; i < 60; i++)
            this.maxSpeed = (this.maxSpeed + MAX_GAS) * AIR_RESISTANCE;
    }

    /**
     * Moves the car along a given track and tracks its progress.
     * @param track the track to drive along
     */
    public void drive(Track track)
    {
        if(!finished)
        {
            controller.update(calculateSensorValues(track));
            updateCarPhysics();
            trackProgress(track);
        }
    }

    /**
     *  Updates the cars position and velocity according to the controllers gas and steering values.
     */
    private void updateCarPhysics()
    {
        // The amount of gas and steering
        double gas = MAX_GAS * controller.getGas();
        double steering = MAX_STEERING * controller.getSteering() * (gas < 0 ? -1 : 1);

        // Calculates the cars speed
        double speed = Math.sqrt(xVel * xVel + yVel * yVel);

        // Change in angular velocity is dependant on the cars speed.
        double steeringTraction = Math.sin(Math.min(1.0, speed / maxSpeed) * Math.PI * ANGULAR_TRACTION_FACTOR);

        // The change in velocity
        double xVelChange = Math.cos(Math.toRadians(rot)) * gas;
        double yVelChange = Math.sin(Math.toRadians(rot)) * gas;

        // Angle between current velocity and the change in velocity decides the traction in the new direction
        double traction = TRACTION_FACTOR +
                Math.abs((xVelChange * xVel + yVelChange * yVel) / speed / gas) * (1.0 - TRACTION_FACTOR);

        // Full traction if no gas or movement
        if(speed == 0 || gas == 0)
            traction = 1;

        // Change in angular velocity is calculated and applied
        rotVel = rotVel * AIR_RESISTANCE + steering * steeringTraction;
        rot += rotVel;
        if(rot > 360) rot -= 360;
        if(rot < 0) rot += 360;

        // Change in translational velocity is calculated and applied
        xVel = (xVel + xVelChange * traction) * AIR_RESISTANCE;
        yVel = (yVel + yVelChange * traction) * AIR_RESISTANCE;

        // Car is moved
        xPos += xVel;
        yPos += yVel;
    }

    /**
     * Tracks the progress of the car along the given track.
     * The car is "finished" if it drives of track, uses too long time in on spot or reaches the tracks end.
     * @param track the track to track the progress along
     */
    private void trackProgress(Track track)
    {
        Track.WayPoint nextWayPoint = track.getWayPoints().get(wayPointsPassed);
        Track.WayPoint lastWayPoint = track.getWayPoints().get(Math.max(0, wayPointsPassed - 1));

        double xDelta = nextWayPoint.x - xPos;
        double yDelta = nextWayPoint.y - yPos;
        double distanceToNextWayPoint = Math.sqrt(xDelta * xDelta + yDelta * yDelta);

        currentFitness = acquiredFitness + (lastWayPoint.distanceToNextWayPoint - distanceToNextWayPoint);

        if(distanceToNextWayPoint < nextWayPoint.trackWidth)
        {
            wayPointsPassed++;
            updatesSinceLastProgress = 0;
            acquiredFitness += lastWayPoint.distanceToNextWayPoint;

            // End of track check
            if(wayPointsPassed >= track.getWayPoints().size())
            {
                finished = true;
                acquiredFitness += CarBreeder.claimReward();
            }
        }

        // Car is considered outside of track if any of the sensor values are inside the car.
        for (double sensorValue : sensorValues)
            if (sensorValue <= (width / 3) / SENSOR_STICK_LENGTH)
                finished = true;

        if(updatesSinceLastProgress >= UPDATES_WITHOUT_PROGRESS)
            finished = true;

        updatesSinceLastProgress++;
    }

    /**
     * Calculates a normalized distance to the tracks "wall" along all sensor-sticks.
     * @param track the track to use for calculation
     * @return an array containing a value for each sensor in the range from 0.0 to 1.0
     */
    private double[] calculateSensorValues(Track track)
    {
        for(int i = 0; i < sensorPoints.length; i++)
        {
            double angle = rot - SENSOR_ANGLE_RANGE + (i + 0.5) * (2 * SENSOR_ANGLE_RANGE / sensorPoints.length);

            sensorPoints[i][0] = xPos + Math.cos(Math.toRadians(angle)) * SENSOR_STICK_LENGTH;
            sensorPoints[i][1] = yPos + Math.sin(Math.toRadians(angle)) * SENSOR_STICK_LENGTH;
            sensorPoints[i][2] = xPos;
            sensorPoints[i][3] = yPos;
            sensorValues[i] = 1.0;

            // Checks for intersection with the past, current and next track segment
            for(int j = wayPointsPassed - 1; j < wayPointsPassed + 1; j++)
            {
                if(j - 1 >= 0 && j < track.getWayPoints().size())
                {
                    Track.WayPoint wayPoint0 = track.getWayPoints().get(j);
                    Track.WayPoint wayPoint1 = track.getWayPoints().get(j - 1);

                    double[] leftIntersection = lineIntersection(xPos, yPos, sensorPoints[i][0], sensorPoints[i][1],
                            wayPoint1.x - wayPoint1.xNormal, wayPoint1.y - wayPoint1.yNormal,
                            wayPoint0.x - wayPoint0.xNormal, wayPoint0.y - wayPoint0.yNormal);

                    double[] rightIntersection = lineIntersection(xPos, yPos, sensorPoints[i][0], sensorPoints[i][1],
                            wayPoint1.x + wayPoint1.xNormal, wayPoint1.y + wayPoint1.yNormal,
                            wayPoint0.x + wayPoint0.xNormal, wayPoint0.y + wayPoint0.yNormal);

                    double[] intersection = (leftIntersection != null) ? leftIntersection : rightIntersection;

                    if(intersection != null)
                    {
                        double xDelta = intersection[0] - xPos;
                        double yDelta = intersection[1] - yPos;
                        double dist = Math.sqrt(xDelta * xDelta + yDelta * yDelta);
                        sensorValues[i] = dist / SENSOR_STICK_LENGTH;
                        sensorPoints[i][2] = intersection[0];
                        sensorPoints[i][3] = intersection[1];
                    }
                }
            }
        }
        return sensorValues;
    }

    /**
     * Finds and returns the intersection point between two line segments.
     * Based on code from the book "Tricks of the Windows Game Programming Gurus" by Andrè LaMothe
     * @param x0 x in first point in first line
     * @param y0 y in first point in first line
     * @param x1 x in second point in first line
     * @param y1 y in second point in first line
     * @param x2 x in first point in second line
     * @param y2 y in first point in second line
     * @param x3 x in second point in second line
     * @param y3 x in second point in second line
     * @return an array containing the x and y coordinate of the intersection point. Null if no intersection.
     */
    private double[] lineIntersection(double x0, double y0, double x1, double y1,
                                      double x2, double y2, double x3, double y3)
    {
        double s1x = x1 - x0;
        double s1y = y1 - y0;
        double s2x = x3 - x2;
        double s2y = y3 - y2;

        double s = (-s1y * (x0 - x2) + s1x * (y0 - y2)) / (-s2x * s1y + s1x * s2y);
        double t = ( s2x * (y0 - y2) - s2y * (x0 - x2)) / (-s2x * s1y + s1x * s2y);

        if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
            return new double[] {x0 + (t * s1x), y0 + (t * s1y)};
        else
            return null;
    }

    /**
     * Renders the car and its sensors.
     * @param camera the camera to be used
     * @param renderSensors whether or not the sensors should be rendered
     */
    public void render(Camera camera, boolean renderSensors)
    {
        GraphicsContext gc = camera.getGraphicsContext();
        camera.startCapture();

        double xCar = xPos - (width / 2);
        double yCar = yPos - (length / 2);
        double wheelRot = controller.getSteering() * 35;

        for(int i = 0; renderSensors && i < sensorPoints.length && !finished; i++)
        {
            gc.setLineWidth(1.0);
            gc.setStroke(Color.gray(0.4,0.5f));
            gc.strokeLine(xPos, yPos, sensorPoints[i][0], sensorPoints[i][1]);
            gc.setFill(Color.RED);
            gc.fillRoundRect(sensorPoints[i][2] - 3, sensorPoints[i][3] - 3, 6, 6, 6, 5);
        }

        // Rotation
        gc.transform(new Affine(new Rotate(rot + 90, xPos, yPos)));

        // Wheels
        //gc.setEffect(dropShadow);
        gc.setFill(Color.rgb(50, 50, 50, finished ? 0.1 : 1.0));
        gc.transform(new Affine(new Rotate(wheelRot, xCar - 2 + (width + 4) / 2, yCar + 5 + 4)));
        gc.fillRect(xCar - 2, yCar + 5, width + 4, 8);
        gc.transform(new Affine(new Rotate(-wheelRot, xCar - 2 + (width + 4) / 2, yCar + 5 + 4)));
        gc.fillRect(xCar - 2, yCar + length - 13, width + 4, 8);

        // Car
        gc.setFill(Color.color(carColor.getRed(), carColor.getGreen(), carColor.getBlue(), finished ? 0.2 : 1.0));
        gc.fillRoundRect(xCar, yCar, width, length, 8, 8);
        gc.setEffect(null);

        // Windows
        gc.setFill(Color.rgb(124, 213, 255, finished ? 0.2 : 1.0));
        gc.fillRoundRect(xCar + 1, yCar + 12, width - 2, 8, 3, 3);
        gc.fillRoundRect(xCar + 1, yCar + 40, width - 2, 2, 3, 3);

        // Rotates the scene back to normal
        gc.transform(new Affine(new Rotate(-(rot + 90), xPos, yPos)));

        camera.endCapture();
    }

    /**
     * Resets the cars position and progress.
     */
    public void reset()
    {
        xPos = yPos = 0;
        xVel = yVel = rotVel = 0;
        rot = 180;
        wayPointsPassed = 0;
        acquiredFitness = 0;
        updatesSinceLastProgress = 0;
        finished = false;
    }

    /**
     * @return the cars horizontal position
     */
    public double getPosX()
    {
        return xPos;
    }

    /**
     * @return the cars vertical position
     */
    public double getPosY()
    {
        return yPos;
    }

    /**
     * @return whether or not the car is finished driving
     */
    public boolean isFinished()
    {
        return finished;
    }

    /**
     * @return the current fitness score
     */
    public double getFitness()
    {
        return currentFitness;

    }

    /**
     * @return the cars controller
     */
    public CarController getController()
    {
        return controller;
    }

    /**
     * Used to sort the cars.
     * @param car the car to compare with
     * @return the difference in fitness
     */
    @Override
    public int compareTo(Car car)
    {
        return (int)(car.acquiredFitness - acquiredFitness);
    }
}
