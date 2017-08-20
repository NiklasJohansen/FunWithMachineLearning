package projects.cardriver.entities;

import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import projects.cardriver.app.Camera;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * This class contains a list of way points resembling a track.
 * The track is created randomly at instantiation and rendered as a road for the cars to drive along.
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public class Track
{
    private static final double MAX_WAY_POINT_LENGTH = 250;
    private static final double MIN_WAY_POINT_LENGTH = 150;
    private static final double MAX_TRACK_WIDTH      = 200;
    private static final double MIN_TRACK_WIDTH      = 150;
    private static final double MAX_TURN_ANGLE       =  45;
    private static final double START_ANGLE          = 180;

    private List<WayPoint> wayPoints;
    private double trackLength;
    private Random random;

    /**
     * Creates a random track with the given length.
     * @param length the length of the track
     */
    public Track(float length)
    {
        this(length, (long)(Math.random() * Long.MAX_VALUE));
    }

    /**
     * Creates a random track from a given seed and length
     * @param length the track length
     * @param seed the random track seed
     */
    public Track(float length, long seed)
    {
        this.trackLength = length;
        this.random = new Random(seed);
        this.wayPoints = new ArrayList<>();
        this.wayPoints.add(new WayPoint(0, 0));

        double totalLength = 0;
        double lastAngle = START_ANGLE;
        while(totalLength < length)
        {
            double dist = MIN_WAY_POINT_LENGTH + random.nextFloat() * (MAX_WAY_POINT_LENGTH - MIN_WAY_POINT_LENGTH);
            totalLength += dist;

            double angle = lastAngle + (random.nextDouble() * 2 * MAX_TURN_ANGLE) - MAX_TURN_ANGLE;
            lastAngle = angle;

            WayPoint lastWayPoint = wayPoints.get(wayPoints.size() - 1);

            double xNew = lastWayPoint.x + Math.cos(Math.toRadians(angle)) * dist;
            double yNew = lastWayPoint.y + Math.sin(Math.toRadians(angle)) * dist;

            wayPoints.add(new WayPoint(xNew, yNew));
            lastWayPoint.addDataAboutNextWayPoint(angle, dist);
        }
        System.out.println("Track seed: " + seed);
    }

    /**
     * Renders the track to screen.
     * @param camera the camera to use
     */
    public void render(Camera camera)
    {
        GraphicsContext gc = camera.getGraphicsContext();

        gc.setFill(Color.LIGHTGREEN);
        gc.fillRect(0, 0, camera.getViewportWidth(), camera.getViewportHeight());

        camera.startCapture();

        for(int i = 0; i < wayPoints.size() - 2; i++)
        {
            WayPoint p0 = wayPoints.get(i);
            WayPoint p1 = wayPoints.get(i + 1);

            double[] xPoints = {p0.x - p0.xNormal, p1.x - p1.xNormal, p1.x + p1.xNormal, p0.x + p0.xNormal};
            double[] yPoints = {p0.y - p0.yNormal, p1.y - p1.yNormal, p1.y + p1.yNormal, p0.y + p0.yNormal};

            // Road
            gc.setFill(Color.LIGHTGRAY);
            gc.fillPolygon(xPoints, yPoints, 4);

            // Side lines
            gc.setLineWidth(3.0);
            gc.setStroke(Color.DARKGRAY);
            gc.strokeLine(p0.x - p0.xNormal, p0.y - p0.yNormal, p1.x - p1.xNormal, p1.y - p1.yNormal);
            gc.strokeLine(p0.x + p0.xNormal, p0.y + p0.yNormal, p1.x + p1.xNormal, p1.y + p1.yNormal);

            // Normals
            //gc.setStroke(Color.RED);
            //gc.strokeLine(p0.x - p0.xNormal, p0.y - p0.yNormal, p0.x + p0.xNormal, p0.y + p0.yNormal);
        }

        // White dashed center line
        gc.beginPath();
        gc.setLineWidth(4);
        gc.setLineDashes(30);
        gc.setLineDashOffset(10);
        gc.setStroke(Color.WHITE);
        gc.moveTo(wayPoints.get(0).x, wayPoints.get(0).y);
        for(int i = 0; i < wayPoints.size() - 1; i++)
            gc.lineTo(wayPoints.get(i).x, wayPoints.get(i).y);
        gc.stroke();
        gc.setLineDashes(0);

        camera.endCapture();
    }

    /**
     * @return the tracks way points.
     */
    public List<WayPoint> getWayPoints()
    {
        return wayPoints;
    }

    /**
     * @return the track length
     */
    public double getLength()
    {
        return trackLength;
    }

    /**
     * This class holds information about every way point along the track.
     */
    public class WayPoint
    {
        public double x;
        public double y;
        public double xNormal;
        public double yNormal;
        public double trackWidth;
        public double distanceToNextWayPoint;

        /**
         * Sets the points position and a random track width.
         * @param x the horizontal position
         * @param y the vertical position
         */
        private WayPoint(double x, double y)
        {
            this.x = x;
            this.y = y;
            this.trackWidth = MIN_TRACK_WIDTH + random.nextDouble() * (MAX_TRACK_WIDTH - MIN_TRACK_WIDTH);
        }

        /**
         * Sets information about the next way point along the track.
         * @param angleToNextWayPoint the angle to the next way point relative to horizontal axis
         * @param distanceToNextWayPoint the distance to the next way point
         */
        private void addDataAboutNextWayPoint(double angleToNextWayPoint, double distanceToNextWayPoint)
        {
            this.distanceToNextWayPoint = distanceToNextWayPoint;
            this.xNormal = Math.cos(Math.toRadians(angleToNextWayPoint + 90)) * trackWidth / 2;
            this.yNormal = Math.sin(Math.toRadians(angleToNextWayPoint + 90)) * trackWidth / 2;
        }
    }
}
