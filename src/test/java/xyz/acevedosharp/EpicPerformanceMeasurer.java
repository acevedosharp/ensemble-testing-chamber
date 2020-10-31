package xyz.acevedosharp;

import ai.libs.hasco.core.events.HASCOSolutionEvent;
import ai.libs.jaicore.components.model.ComponentInstance;
import ai.libs.jaicore.components.model.RefinementConfiguredSoftwareConfigurationProblem;

import java.io.*;
import java.net.URISyntaxException;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import ai.libs.hasco.builder.HASCOBuilder;
import ai.libs.hasco.builder.forwarddecomposition.HASCOViaFD;
import ai.libs.hasco.core.HASCOSolutionCandidate;

import ai.libs.jaicore.components.serialization.ComponentSerialization;
import ai.libs.jaicore.logging.LoggerUtil;
import ai.libs.jaicore.ml.classification.loss.dataset.EClassificationPerformanceMeasure;
import ai.libs.jaicore.ml.core.dataset.serialization.ArffDatasetAdapter;
import ai.libs.jaicore.ml.core.evaluation.evaluator.SupervisedLearnerExecutor;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.mlplan.core.MLPlan;
import ai.libs.mlplan.weka.MLPlanWekaBuilder;
import org.api4.java.ai.ml.classification.singlelabel.evaluation.ISingleLabelClassification;
import org.api4.java.ai.ml.core.dataset.serialization.DatasetDeserializationFailedException;
import org.api4.java.ai.ml.core.dataset.splitter.SplitFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.execution.ILearnerRunReport;
import org.api4.java.ai.ml.core.evaluation.execution.LearnerExecutionFailedException;
import org.api4.java.algorithm.Timeout;

import org.api4.java.algorithm.exceptions.AlgorithmException;
import org.api4.java.algorithm.exceptions.AlgorithmExecutionCanceledException;
import org.junit.Test;

import static org.junit.Assert.*;


import java.io.File;
import java.util.stream.Collectors;

public class EpicPerformanceMeasurer {

    private static final Random random = new Random(0);
    private static final List<String> DATASET_NAMES = List.of("iris.arff");
    private static final Timeout TIMEOUT_15S = new Timeout(15, TimeUnit.SECONDS);

    @Test
    public void epicRunner() throws IOException, InterruptedException, TimeoutException, AlgorithmExecutionCanceledException, AlgorithmException, URISyntaxException, DatasetDeserializationFailedException, SplitFailedException {
        /* Load datasets */
        List<ILabeledDataset<?>> datasets = DATASET_NAMES.stream().map(
                s -> {
                    try {
                        File dsFile = new File(this.getClass().getClassLoader().getResource(s).toURI());
                        return ArffDatasetAdapter.readDataset(dsFile);
                    } catch (DatasetDeserializationFailedException | URISyntaxException e) {
                        e.printStackTrace();
                    }
                    return null; // shouldn't happen
                }
        ).collect(Collectors.toList());

        for (ILabeledDataset<?> el : datasets) {
            assertNotNull(el);
        }


        /* Create HASCO */
        String reqInterface = "AbstractClassifier";
        File componentFile = new File(this.getClass().getClassLoader().getResource("weka-classifiers.json").toURI());
        RefinementConfiguredSoftwareConfigurationProblem<Double> problem =
                new RefinementConfiguredSoftwareConfigurationProblem<>(componentFile, reqInterface, n -> random.nextDouble() * 0.1);

        HASCOViaFD<Double> hasco = HASCOBuilder.get()
                .withBestFirst()
                .withRandomCompletions()
                .withProblem(problem)
                .withDefaultParametrizationsFirst()
                .getAlgorithm();

        hasco.setTimeout(TIMEOUT_15S);


        /* Run & benchmark HASCO on DATASETS - store results (in memory) */
        Map<String, BenchmarkResult> hascoResults = new HashMap<>();
        ComponentInstance solution = hasco.nextSolutionCandidate().getComponentInstance();

        /* Create, Run & benchmark MLPlan on DATASETS - store results (in memory) */
        Map<String, BenchmarkResult> mlPlanResults = new HashMap<>();

        for (int i = 0; i < datasets.size(); i++) {
            ILabeledDataset<?> el = datasets.get(i);

            List<ILabeledDataset<?>> split = SplitterUtil.getLabelStratifiedTrainTestSplit(el, new Random(42), .7); // TODO: parametrize seed
            MLPlan<IWekaClassifier> mlplan = new MLPlanWekaBuilder().withNumCpus(4).withTimeOut(TIMEOUT_15S).withDataset(split.get(0)).build();

            long start;
            IWekaClassifier optimizedClassifier;
            try {
                start = System.currentTimeMillis();
                optimizedClassifier = mlplan.call();
                long trainTime = System.currentTimeMillis() - start;
                System.out.println("Finished build of the classifier. Training time was " + trainTime + "s");
                System.out.println("Chosen model is: " + mlplan.getSelectedClassifier());

                /* evaluate solution produced by mlplan */
                SupervisedLearnerExecutor executor = new SupervisedLearnerExecutor();
                ILearnerRunReport report = executor.execute(optimizedClassifier, split.get(1));
                double errorRate = EClassificationPerformanceMeasure.ERRORRATE.loss(report.getPredictionDiffList().getCastedView(Integer.class, ISingleLabelClassification.class));
                System.out.println("Error Rate of the solution produced by ML-Plan: " +
                        errorRate +
                        ". Internally believed error was " +
                        mlplan.getInternalValidationErrorOfSelectedClassifier()
                );

                mlPlanResults.put(DATASET_NAMES.get(i), new BenchmarkResult(trainTime, errorRate));
            } catch (NoSuchElementException | LearnerExecutionFailedException e) {
                System.out.println("Building the classifier failed: " + e.getMessage());
            }
        }


        /* Compare and persist results */


        // collect solutions
        // while (hasco.hasNext()) {
        //     while (hasco.hasNext()) {
        //         IAlgorithmEvent event = hasco.nextWithException();
        //         if (event instanceof HASCOSolutionEvent) {
        //             @SuppressWarnings("unchecked")
        //             ComponentInstance solution = ((HASCOSolutionEvent<Double>) event).getSolutionCandidate().getComponentInstance();
        //             String serializedSolution = new ComponentSerialization().serialize(solution).toString();
        //             System.out.println(serializedSolution);
        //         }
        //     }
        // }
    }
}
