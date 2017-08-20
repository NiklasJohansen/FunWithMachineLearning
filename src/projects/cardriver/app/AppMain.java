package projects.cardriver.app;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.AnchorPane;
import javafx.stage.Stage;

/**
 * The main class of the application.
 * Loads the FXML document and sets up the scene and primary stage.
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public class AppMain extends Application
{
    @Override
    public void start(Stage primaryStage) throws Exception
    {
        FXMLLoader loader = new FXMLLoader();
        loader.setLocation(getClass().getResource("userinterface.fxml"));

        AnchorPane root = loader.load();
        Scene scene = new Scene(root, root.getPrefWidth(), root.getPrefHeight());

        primaryStage.setTitle("Car Driver Demo - github.com/NiklasJohansen/FunWithMachineLearning");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    public static void main(String[] args)
    {
        launch(args);
    }
}
