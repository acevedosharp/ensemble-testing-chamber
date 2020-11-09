package xyz.acevedosharp;

import ai.libs.hasco.core.events.HASCOSolutionEvent;
import ai.libs.jaicore.components.model.ComponentInstance;
import ai.libs.jaicore.components.model.RefinementConfiguredSoftwareConfigurationProblem;

import java.io.*;
import java.net.URISyntaxException;
import java.util.*;
import java.util.concurrent.TimeUnit;

import ai.libs.hasco.builder.HASCOBuilder;
import ai.libs.hasco.builder.forwarddecomposition.HASCOViaFD;
import ai.libs.hasco.core.HASCOSolutionCandidate;

import org.api4.java.algorithm.Timeout;

import org.api4.java.algorithm.events.IAlgorithmEvent;
import org.api4.java.algorithm.exceptions.AlgorithmException;
import org.api4.java.algorithm.exceptions.AlgorithmExecutionCanceledException;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import org.api4.java.common.attributedobjects.IObjectEvaluator;
import org.junit.Test;

import java.io.File;
import java.util.stream.Collectors;

@SuppressWarnings({"UnusedLabel", "unchecked", "ConstantConditions", "rawtypes", "ForLoopReplaceableByForEach"})
public class EpicPerformanceMeasurer {

    private static final List<String> DATASET_NAMES = Arrays.asList("datasets/iris.arff");
    private static final Timeout TIMEOUT = new Timeout(16, TimeUnit.SECONDS);

    @Test
    public void epicRunner() throws IOException, URISyntaxException {


            /*MLPlan: {
                List<ILabeledDataset<?>> split = SplitterUtil.getLabelStratifiedTrainTestSplit(ds, new Random(42), .7);  //TODO: parametrize seed
                MLPlan<IWekaClassifier> mlplan = new MLPlanWekaBuilder().withNumCpus(2).withTimeOut(TIMEOUT_15S).withDataset(split.get(0)).build();

                long start;
                IWekaClassifier optimizedClassifier;
                try {
                    start = System.currentTimeMillis();
                    optimizedClassifier = mlplan.call();
                    long trainTime = System.currentTimeMillis() - start;
                    System.out.println("Finished build of the classifier. Training time was " + trainTime + "s");
                    System.out.println("Chosen model is: " + mlplan.getSelectedClassifier());

                    *//* evaluate solution produced by mlplan *//*
                    SupervisedLearnerExecutor executor = new SupervisedLearnerExecutor();
                    ILearnerRunReport report = executor.execute(optimizedClassifier, split.get(1));
                    double errorRate = EClassificationPerformanceMeasure.ERRORRATE.loss(report.getPredictionDiffList().getCastedView(Integer.class, ISingleLabelClassification.class));
                    System.out.println("Error Rate of the solution produced by ML-Plan: " +
                            errorRate +
                            ". Internally believed error was " +
                            mlplan.getInternalValidationErrorOfSelectedClassifier()
                    );

                    results.add(new BenchmarkResult(
                            DATASET_NAMES.get(i),
                            Algos.MLPLAN,
                            trainTime,
                            errorRate,
                            mlplan.getSelectedClassifier().toString()
                    ));
                } catch (NoSuchElementException | LearnerExecutionFailedException e) {
                    System.out.println("Building the classifier failed: " + e.getMessage());
                }
            }*/





        /* Create, Run & benchmark MLPlan on DATASETS - store results (in memory) */
        Map<String, BenchmarkResult> mlPlanResults = new HashMap<>();

    }

    @Test
    public void runWithHasco() throws URISyntaxException, IOException, AlgorithmTimeoutedException {
        List<File> datasets = datasetsAsFiles();

        for (int i = 0; i < datasets.size(); i++) {
            File dsFile = datasets.get(i);

            String reqInterface = "EpicEnsemble";
            File componentFile = new File(this.getClass().getClassLoader().getResource("search-space/ensemble-configuration.json").toURI());

            IObjectEvaluator<ComponentInstance, Double> evaluator = new EpicEnsembleEvaluator(dsFile);

            RefinementConfiguredSoftwareConfigurationProblem<Double> problem =
                    new RefinementConfiguredSoftwareConfigurationProblem(componentFile, reqInterface, evaluator);

            HASCOViaFD<Double> hasco = HASCOBuilder.get()
                    .withBestFirst()
                    .withRandomCompletions()
                    .withProblem(problem)
                    .withDefaultParametrizationsFirst()
                    .getAlgorithm();

            hasco.setTimeout(TIMEOUT);


            ArrayList<HASCOSolutionCandidate<Double>> solutions = new ArrayList<>();

            while (hasco.hasNext()) {
                try {
                    IAlgorithmEvent e = hasco.nextWithException();
                    if (e instanceof HASCOSolutionEvent) {
                        @SuppressWarnings("unchecked")
                        HASCOSolutionCandidate<Double> s = ((HASCOSolutionEvent<Double>) e).getSolutionCandidate();
                        solutions.add(s);
                    }
                } catch (AlgorithmTimeoutedException ex) {
                    break;
                } catch (InterruptedException | AlgorithmExecutionCanceledException | AlgorithmException exx) {
                    exx.printStackTrace();
                }
            }

            System.out.println(solutions.size());

            //System.out.println("sol.: " + solution.getComponentInstance().toString() + " with score: " + solution.getScore());
        }
    }

    @Test
    public void runWithMLPlan() {
    }

    private List<File> datasetsAsFiles() {
        return DATASET_NAMES.stream().map(
                s -> {
                    try {
                        return new File(this.getClass().getClassLoader().getResource(s).toURI());
                    } catch (URISyntaxException e) {
                        e.printStackTrace();
                    }
                    return null;  // shouldn't happen
                }
        ).collect(Collectors.toList());
    }
}
