package projects.cardriver.app;

import neuralnetwork.NeuralNetwork;
import neuralnetwork.training.GeneticAlgorithm;
import projects.cardriver.controllers.NNCarController;
import projects.cardriver.controllers.UserCarController;
import projects.cardriver.entities.Car;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * This class is used to create a new generation of cars based upon how well the current
 * generation did along the track. Cars with a higher fitness score have higher chance of
 * producing offspring. Chance based breeding creates diversity among the population,
 * which in turn creates cars more fit to overcome challenging parts of a track. Only
 * breeding the elite cars tends to cause a rapid convergence on local minimas, where cars
 * get stuck early in the track without any further progress. This class provides a combined
 * approach where fitness determines the breeding chance and (if activated) the top car gets
 * bred with a random car from the elite group. Uses the {@link GeneticAlgorithm} class in the
 * {@link neuralnetwork} package.
 *
 * @author Niklas Johansen
 * @version 1.0
 */
public class CarBreeder
{
    private static final double MUTATION_PROBABILITY = 20;
    private static final double ELITE_GROUP_PERCENTAGE = 10;
    private static final double INIT_SPEED_REWARD = 100000;
    private static double speedReward;

    private GeneticAlgorithm geneticAlgorithm;
    private int generationCounter = 1;

    public CarBreeder()
    {
        this.geneticAlgorithm = new GeneticAlgorithm();
    }

    /**
     * Creates a new generation by breeding cars from the current generation.
     * @param currentCars the list of cars in the current generation
     * @param nCars the number of odspring cars in the new generation
     * @param eliteBreading whether or not the top car should be bred with another car from the elite group.
     * @return
     */
    public List<Car> getNextGeneration(List<Car> currentCars, int nCars, boolean eliteBreading)
    {
        // No point in breeding without someone to breed with
        if(currentCars.size() == 1)
        {
            currentCars.get(0).reset();
            return currentCars;
        }

        List<Car> nextGenCars = new ArrayList<>();

        // Removes user controlled cars from the breeding list
        handleUserControlledCars(currentCars, nextGenCars);

        double fitnessSum = 0;
        for(Car car : currentCars)
            fitnessSum += car.getFitness();

        for(int i = 0; i < nCars && currentCars.size() > 1; i++) // nCars / 2
        {
            double fixedPoint1 = Math.random() * fitnessSum;
            double fixedPoint2 = Math.random() * fitnessSum;

            Car mother = null;
            Car father = null;

            double sum = 0;
            for(Car car : currentCars)
            {
                sum += car.getFitness();
                if(mother == null && sum > fixedPoint1) mother = car;
                if(father == null && sum > fixedPoint2) father = car;
            }

            if(mother != father)
            {
                NeuralNetwork offspring = geneticAlgorithm.breed(
                        ((NNCarController) mother.getController()).getNeuralNetwork(),
                        ((NNCarController) father.getController()).getNeuralNetwork(),
                        MUTATION_PROBABILITY);

                nextGenCars.add(new Car(0, 0, new NNCarController(offspring)));
            }
            else i--;
        }

        // Elite breeding
        if(eliteBreading && currentCars.size() > 1)
        {
            Collections.sort(currentCars);

            Car mother = currentCars.get(0);
            Car father = currentCars.get(1 +
                    (int)(Math.random() * (Math.min(currentCars.size() - 2, nCars * (ELITE_GROUP_PERCENTAGE / 100)))));

            NeuralNetwork offspring = geneticAlgorithm.breed(
                    ((NNCarController) mother.getController()).getNeuralNetwork(),
                    ((NNCarController) father.getController()).getNeuralNetwork(),
                    MUTATION_PROBABILITY);

            nextGenCars.add(new Car(0, 0, new NNCarController(offspring)));
        }

        speedReward = INIT_SPEED_REWARD;
        generationCounter++;
        currentCars.clear();
        return nextGenCars;
    }

    /**
     * Cars that finish the whole track get a fitness bonus. Every car gets half of the remaining reward.
     * Faster cars will then be rewarded more, giving them a higher chance of producing offspring.
     * @return the current reward
     */
    public static double claimReward()
    {
        return (speedReward *= 0.5);
    }

    /**
     * Moves the user controlled cars to the next generation,
     * @param currentCars the current generation of cars
     * @param nextGenCars the next generation of cars
     */
    private void handleUserControlledCars(List<Car> currentCars, List<Car> nextGenCars)
    {
        for(int i = 0; i < currentCars.size(); i++)
        {
            Car car = currentCars.get(i);
            if (car.getController() instanceof UserCarController)
            {
                car.reset();
                nextGenCars.add(car);
                currentCars.remove(i--);
            }
        }
    }

    /**
     * Resets the generation counter.
     */
    public void resetGenerationCount()
    {
        generationCounter = 1;
    }

    /**
     * @return the current generation
     */
    public int getGeneration()
    {
        return generationCounter;
    }
}
