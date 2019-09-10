package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer
 * or more than 15 rings.
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class AbaloneTest {
    private static Instance[] testInstances = initializeTestInstances();
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 28, hiddenLayer = 5, outputLayer = 1, trainingIterations = 1000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E8, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);


            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < instances.length; j++) {
                networks[i].setInputValues(testInstances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(instances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                if (actual == 1 && predicted == 0) {

                }

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        }

        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        double[] trainingErrorArray = new double[trainingIterations];
        double[] testErrorArray = new double[trainingIterations];

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            double testError = 0;

            for (int k = 0; k < testInstances.length; k++) {
                Instance testOutput = testInstances[k].getLabel(), testExample = new Instance(network.getOutputValues());
                testExample.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                testError += measure.value(testOutput, testExample);
            }

            trainingErrorArray[i] = error / ((double) 4);
            testErrorArray[i] = testError;

            System.out.println("Train error: " + df.format(error / ((double) 4)));
            System.out.println("Test error: " + df.format(testError));
        }

        System.out.println();
        System.out.println("Train Error: ");

        for (int i = 0; i < trainingErrorArray.length; i++) {
            System.out.print(trainingErrorArray[i] + ", ");
        }

        System.out.println();
        System.out.println("Test Error: ");

        for (int i = 0; i < testErrorArray.length; i++) {
            System.out.print(testErrorArray[i] + ", ");
        }
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[7999][][];
        // double[][][] testAttributes = new double[2000][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("creditCard_Train_Binary.csv")));

            br.readLine();

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[28]; // 28 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 28; j++)  {
                    attributes[i][0][j] = Integer.parseInt(scan.next());
                }

                attributes[i][1][0] = Integer.parseInt(scan.next());

            }

        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        // Instance[] testInstances = new Instance[testAttributes.length];
        //
        // for(int i = 0; i < testInstances.length; i++) {
        //     testInstances[i] = new Instance(testAttributes[i][0]);
        //     testInstances[i].setLabel(new Instance(testAttributes[i][1][0]));
        // }
        //
        // this.testInstances = testInstances;

        return instances;
    }

    private static Instance[] initializeTestInstances() {
        double[][][] testAttributes = new double[2000][][];

        try {
            BufferedReader br1 = new BufferedReader(new FileReader(new File("creditCard_Test_Binary.csv")));

            br1.readLine();

            for(int i = 0; i < testAttributes.length; i++) {
                Scanner scan = new Scanner(br1.readLine());
                scan.useDelimiter(",");

                testAttributes[i] = new double[2][];
                testAttributes[i][0] = new double[28]; // 28 attributes
                testAttributes[i][1] = new double[1];

                for(int j = 0; j < 28; j++)  {
                    testAttributes[i][0][j] = Integer.parseInt(scan.next());
                }

                testAttributes[i][1][0] = Integer.parseInt(scan.next());

            }
        } catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] testInstances = new Instance[testAttributes.length];

        for(int i = 0; i < testInstances.length; i++) {
            testInstances[i] = new Instance(testAttributes[i][0]);
            testInstances[i].setLabel(new Instance(testAttributes[i][1][0]));
        }

        return testInstances;

    }

}
