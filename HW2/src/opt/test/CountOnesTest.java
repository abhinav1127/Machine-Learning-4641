package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
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
public class CountOnesTest {
    /** The n value */
    private static final int N = 80;

    public static void main(String[] args) {

        int[] iterArray = {50, 100, 500, 1000, 2000, 5000, 7500, 10000, 12000, 15000, 20000, 25000};
        long startTime;
        long endTime;
        long timeElapsed;

        for (int j = 0; j < iterArray.length; j++) {
            int[] ranges = new int[N];
            Arrays.fill(ranges, 2);
            EvaluationFunction ef = new CountOnesEvaluationFunction();
            Distribution odd = new DiscreteUniformDistribution(ranges);
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new UniformCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

            startTime = System.nanoTime();
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iterArray[j]);
            fit.train();
            endTime = System.nanoTime();
            timeElapsed = (endTime - startTime);
            System.out.println("Randomized Hill Climbing with " + iterArray[j] + " iterations results in: " + ef.value(rhc.getOptimal()) + " and training time was " + (timeElapsed / 1000000) + " milliseconds.");

            startTime = System.nanoTime();
            SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
            fit = new FixedIterationTrainer(sa, iterArray[j]);
            fit.train();
            endTime = System.nanoTime();
            timeElapsed = (endTime - startTime);
            System.out.println("Simulated Annealing with " + iterArray[j] + " iterations results in: " + ef.value(sa.getOptimal()) + " and training time was " + (timeElapsed / 1000000) + " milliseconds.");

            startTime = System.nanoTime();
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(20, 20, 0, gap);
            fit = new FixedIterationTrainer(ga, iterArray[j]);
            fit.train();
            endTime = System.nanoTime();
            timeElapsed = (endTime - startTime);
            System.out.println("Genetic Algorithm with " + iterArray[j] + " iterations results in: " + ef.value(ga.getOptimal()) + " and training time was " + (timeElapsed / 1000000) + " milliseconds.");

            System.out.println("\n\n\n");
        }
    }
}
