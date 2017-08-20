package projects.cardriver.controllers;

import javafx.scene.input.KeyEvent;

/**
 * This class extends the {@link CarController}.
 * Updates in the gas and steering values are based upon key input.
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public class UserCarController extends CarController
{
    public void ketInput(KeyEvent event, boolean pressed)
    {
        switch (event.getCode())
        {
            case A: super.steering = pressed ? -0.5f : 0; break;
            case D: super.steering = pressed ?  0.5f : 0; break;
            case S: super.gas = pressed ? -0.25f : 0; break;
            case W: super.gas = pressed ? 0.75f : 0; break;
        }
    }

    @Override
    public void update(double[] sensorInputs) {}
}
