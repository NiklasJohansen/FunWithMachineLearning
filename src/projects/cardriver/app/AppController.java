package projects.cardriver.app;

import javafx.scene.control.Alert;
import javafx.scene.input.KeyCode;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.datautils.Utils;
import projects.cardriver.controllers.NNCarController;
import projects.cardriver.controllers.UserCarController;
import projects.cardriver.entities.Car;
import projects.cardriver.entities.Track;
import javafx.animation.AnimationTimer;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.layout.AnchorPane;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.TextAlignment;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * The main JavaFX controller class.
 * Handles the game loop, rendering calls and input events.
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public class AppController
{
    private static final int CARS_PER_GEN = 25;
    private static final int TRACK_LENGTH = 10000;

    @FXML private AnchorPane anchorPane;
    @FXML private Canvas canvas;

    private List<Car> cars;

    private Track track;
    private Camera camera;
    private CarBreeder carBreeder;
    private NetworkGraph networkGraph;
    private UserCarController userCarController;

    private double currentHighestFitness;
    private double highestOverallFitness;
    private int simulationSpeed = 1;
    private int finishedCars;
    private boolean runSimulation;

    /**
     * Called when all FXML elements are loaded.
     * Instantiates local objects, adds event listeners and cars.
     */
    @FXML
    public void initialize()
    {
        this.cars = new ArrayList<>();
        this.track = new Track(TRACK_LENGTH, 1536032139186569216L);
        this.camera = new Camera(canvas);
        this.carBreeder = new CarBreeder();
        this.networkGraph = new NetworkGraph();
        this.runSimulation = true;

        addEventHandlers();
        addAndPreTrainCars(30,0);
        //addUserControlledCar();
    }

    /**
     * Updates the main game logic.
     * Trains the cars by letting them drive, tracking their progress and
     * breeding new generations based upon the cars fitness score.
     */
    private void updateGameLogic()
    {
        currentHighestFitness = 0;
        finishedCars = 0;

        for(Car car : cars)
        {
            car.drive(track);

            if(!car.isFinished() && car.getFitness() > currentHighestFitness)
            {
                currentHighestFitness = car.getFitness();
                highestOverallFitness = Math.max(highestOverallFitness, currentHighestFitness);
                camera.track(car);
            }

            if(car.isFinished())
                finishedCars++;
        }

        if(finishedCars > 0 && finishedCars == cars.size())
            cars = carBreeder.getNextGeneration(cars, CARS_PER_GEN, false);
    }

    /**
     * Renders the cars, the track, the text and the network graph.
     * The car with the highest fitness will be tracked by the camera.
     * If a user controlled car is added, this will be tracked.
     */
    private void renderScene()
    {
        track.render(camera);

        for(Car car : cars)
        {
            car.render(camera, (car == camera.getTrackedCar()));
            if(!car.isFinished() && car.getController() instanceof UserCarController)
                camera.track(car);
        }

        camera.update();

        renderGraphAndInfoText();
    }

    /**
     * Renders the network graph and informational text about the simulation.
     */
    private void renderGraphAndInfoText()
    {
        int graphBottom = 25;
        Car car = camera.getTrackedCar();
        if(car != null && car.getController() instanceof NNCarController)
        {
            networkGraph.render(((NNCarController)car.getController()).getNeuralNetwork(), canvas, 40, 40);
            graphBottom = networkGraph.getHeight() + 40;
        }

        GraphicsContext gc = canvas.getGraphicsContext2D();

        gc.setFill(Color.gray(0,0.75));
        gc.setFont(Font.font(24));
        gc.setTextAlign(TextAlignment.LEFT);

        gc.fillText("Simulation speed: " + (runSimulation ? simulationSpeed : 0),15, graphBottom +  10);
        gc.fillText("Driving cars: "     + (cars.size() - finishedCars),         15, graphBottom +  40);
        gc.fillText("Generation: "       + carBreeder.getGeneration(),           15, graphBottom +  70);
        gc.fillText("Top fitness: "      + (int)highestOverallFitness,           15, graphBottom + 100);
        gc.fillText("Fitness: "          + (int)currentHighestFitness,           15, graphBottom + 130);
    }

    /**
     * Sets up the main game loop and adds event handlers for mouse/key input and window resizing.
     */
    private void addEventHandlers()
    {
        // The main game loop
        new AnimationTimer()
        {
            @Override
            public void handle(long now)
            {
                for(int i = 0; i < simulationSpeed && runSimulation; i++)
                    updateGameLogic();
                renderScene();
            }
        }.start();

        // Event handler for scroll wheel
        canvas.setOnScroll(event -> camera.zoom(event.getDeltaY() / 1000.0));

        // Event handler for all key events
        Platform.runLater(this::addKeyEvents);

        // Event handlers for window resizing
        Platform.runLater(() ->
        {
            anchorPane.getScene().widthProperty().addListener((o, oldVal, newVal) -> canvas.setWidth(newVal.intValue()));
            anchorPane.getScene().heightProperty().addListener((o, oldVal, newVal) -> canvas.setHeight(newVal.intValue()));
        });
    }

    private void  addKeyEvents()
    {
        // Event handler for key presses
        anchorPane.getScene().setOnKeyPressed(event ->
        {
            if(userCarController != null)
                userCarController.ketInput(event, true);

            KeyCode code = event.getCode();
            if(event.isControlDown())
            {
                // Saves the network of the tracked car to a file
                if(code == KeyCode.S)
                {
                    Car bestCar = camera.getTrackedCar();
                    if(bestCar != null && bestCar.getController() instanceof NNCarController)
                        Utils.exportNetwork(((NNCarController) bestCar.getController()).getNeuralNetwork(), "CarDriver");
                }

                // Opens a file chooser to load exported networks
                else if(code == KeyCode.O)
                {
                    List<File> files = (new FileChooser()).showOpenMultipleDialog(null);
                    if(files != null && files.size() > 0)
                        for(File f : files)
                            addCarFromFile(f.getAbsolutePath());
                }

                // Resets all training with new cars
                else if(code == KeyCode.R)
                {
                    cars.clear();
                    carBreeder.resetGenerationCount();
                    addAndPreTrainCars(CARS_PER_GEN, 0);
                }

                // Clears all cars
                else if(code == KeyCode.C)
                {
                    cars.clear();
                }

                // Creates a new and random track
                else if(code == KeyCode.T)
                {
                    for(Car car : cars)
                        car.reset();
                    track = new Track(TRACK_LENGTH);
                }

                // Adds or removes a user controller car
                if(code == KeyCode.U)
                {
                    if(userCarController == null)
                        addUserControlledCar();
                    else
                    {
                        userCarController = null;
                        for(int i = 0; i < cars.size(); i++)
                            if(cars.get(i).getController() instanceof UserCarController)
                                cars.remove(i--);
                    }
                }

                // Fullscreen
                else if(code == KeyCode.F)
                {
                    Stage stage = (Stage) anchorPane.getScene().getWindow();
                    stage.setFullScreen(!stage.isFullScreen());
                }
            }


            // Starts/stops the simulation
            if(code == KeyCode.SPACE)
                runSimulation = !runSimulation;
            // Simulation speed keys
            else if(code == KeyCode.PLUS || code == KeyCode.ADD)
                simulationSpeed++;
            else if(code == KeyCode.MINUS || code == KeyCode.SUBTRACT)
                simulationSpeed = Math.max(1, simulationSpeed - 1);
            else if(code == KeyCode.NUMPAD0 || code == KeyCode.ENTER)
                simulationSpeed = 1;
        });

        // Event handler for key releases
        anchorPane.getScene().setOnKeyReleased(event ->
        {
            if(userCarController != null)
                userCarController.ketInput(event, false);
        });
    }

    /**
     * Adds the first generation of cars.
     * The cars can be pre-trained before the real-time simulation starts.
     * @param nCars the number of cars in the first generation
     * @param nGameLoopsWithPreTraining the number of game loops to pre-train the cars
     */
    private void addAndPreTrainCars(int nCars, int nGameLoopsWithPreTraining)
    {
        for(int i = 0; i < nCars; i++)
            cars.add(new Car(0,0, new NNCarController()));

        if(nGameLoopsWithPreTraining > 0)
        {
            for(double i = 0, topFitness = 0; i < nGameLoopsWithPreTraining; i++)
            {
                updateGameLogic();

                // Prints the progress to the console.
                if(highestOverallFitness > topFitness)
                {
                    topFitness = highestOverallFitness;
                    System.out.println("Gen: " + carBreeder.getGeneration() + "  -  " +
                            (int)(Math.min(1.0, topFitness / track.getLength()) * 100) + " % of track learned");
                }
            }
        }
    }

    /**
     * Enables previously trained cars to be loaded from files.
     * @param filename the file containing the trained neural network
     */
    private void addCarFromFile(String filename)
    {
        NeuralNetwork network = Utils.importNetwork(filename);
        if(network != null)
            this.cars.add(new Car(0,0, new NNCarController(network)));
        else
            (new Alert(Alert.AlertType.ERROR, "The selected file could not be loaded")).showAndWait();
    }

    /**
     * Adds a user controlled car to the list of cars
     */
    private void addUserControlledCar()
    {
        this.userCarController = new UserCarController();
        this.cars.add(new Car(0, 0, userCarController));
    }
}
