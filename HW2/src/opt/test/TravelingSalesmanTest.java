package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

import java.util.concurrent.TimeUnit;

/**
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest {
    /** The n value */
    private static final int N = 50;
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {

        int[] iterArray = {50, 100, 500, 1000, 2000, 5000, 7500, 10000, 12000, 15000, 20000, 25000};
        long startTime;
        long endTime;
        long timeElapsed;

        for (int j = 0; j < iterArray.length; j++) {
            Random random = new Random();
            // create the random points
            double[][] points = new double[N][2];
            for (int i = 0; i < points.length; i++) {
                points[i][0] = random.nextDouble();
                points[i][1] = random.nextDouble();
            }
            // for rhc, sa, and ga we use a permutation based encoding
            TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
            Distribution odd = new DiscretePermutationDistribution(N);
            NeighborFunction nf = new SwapNeighbor();
            MutationFunction mf = new SwapMutation();
            CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

            startTime = System.nanoTime();
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iterArray[j]);
            fit.train();
            endTime = System.nanoTime();
            timeElapsed = (endTime - startTime);
            System.out.println("Randomized Hill Climbing with " + iterArray[j] + " iterations results in: " + ef.value(rhc.getOptimal()) + " and training time was " + (timeElapsed / 1000000) + " milliseconds.");

            startTime = System.nanoTime();
            SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
            fit = new FixedIterationTrainer(sa, iterArray[j]);
            fit.train();
            endTime = System.nanoTime();
            timeElapsed = (endTime - startTime);
            System.out.println("Simulated Annealing with " + iterArray[j] + " iterations results in: " + ef.value(sa.getOptimal()) + " and training time was " + (timeElapsed / 1000000) + " milliseconds.");

            startTime = System.nanoTime();
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
            fit = new FixedIterationTrainer(ga, iterArray[j]);
            fit.train();
            endTime = System.nanoTime();
            timeElapsed = (endTime - startTime);
            System.out.println("Genetic Algorithm with " + iterArray[j] + " iterations results in: " + ef.value(ga.getOptimal()) + " and training time was " + (timeElapsed / 1000000) + " milliseconds.");

            System.out.println("\n\n\n");

        }

    }
}
